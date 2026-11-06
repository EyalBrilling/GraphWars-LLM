import os
import sys
import json
import re
import argparse
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from src.experiments.runner import BenchmarkRunner
from src.engine.backend.physics import TrajectoryResult


SYSTEM_PROMPT = """You are the Trajectory Planner for GraphWars-LLM in a continuous 2D Cartesian plane (x, y in R).
Your goal is to formulate an explicit mathematical function y = f(x) that starts at the shooter's coordinate, navigates around obstacles without colliding, and hits the target within its radius.

ATTEMPT BUDGET & STRATEGY:
- You have a STRICT LIMITED BUDGET of maximum allowed attempts (see attempt_info in payload).
- Think of a strategy that maximizes hit success in the number of attempts you have.
- Each attempt has a fixed output budget shared by your thinking and your answer. If you run out before giving the JSON answer, the attempt counts as a miss (hit_type "no_answer"). A reasonable formula you can refine next attempt beats a perfect one you never write.

INPUT & ATTEMPT HISTORY ANALYSIS:
- You receive the arena geometry (shooter, target, bounds, obstacles), attempt_info budget, and an array of past attempts under 'history'.
- On Attempt 1, 'history' is empty [].
- On Attempt >1, 'history' contains your previous formulas and their exact simulation results (hit_type, hit_coordinate, hit_obstacle, closest_distance_to_target).
- CRITICAL: Carefully inspect the 'hit_coordinate' and 'hit_obstacle' in 'history' to adjust your curve's amplitude, frequency, or vertical shift to clear the obstacles you previously collided with!

CRITICAL MATHEMATICAL & SYNTAX RULES:
1. The formula MUST be an explicit function of 'x' (e.g. "0.05 * x**2 - 3").
2. Use standard Python math syntax:
   - Use '*' for multiplication (e.g., '2 * x', NOT '2x').
   - Use '**' or '^' for exponentiation (e.g., 'x**2').
   - Supported functions: sin, cos, tan, exp, log, sqrt, abs, pi, e.
3. Boundary conditions: |f(x_shooter) - y_shooter| must be <= shooter.radius, otherwise the shot is rejected with hit_type "invalid_start". f(x_target) must reach inside target radius.
4. Formula character length limit: strictly obey constraints.max_formula_characters; longer formulas are rejected with hit_type "invalid_formula".
5. Only numbers, x, pi, e, + - * / ** and the supported function calls are accepted; anything else is rejected with hit_type "invalid_formula".

OUTPUT CONTRACT:
You MUST respond with ONLY a valid JSON object matching this exact schema:
{
  "reasoning": "Detailed step-by-step spatial derivation: 1) Analyze obstacles & gaps, 2) List chosen waypoints (x_k, y_k) with clearance, 3) Derive the explicit formula y = f(x) and verify boundary conditions.",
  "strategy": "arc_over | s_curve | direct | low_tunnel | trigonometric_wave",
  "planned_waypoints": [
    {"x": -20.0, "y": 0.0},
    {"x": 0.0, "y": 8.0},
    {"x": 20.0, "y": 0.0}
  ],
  "formula": "-0.02 * (x + 20) * (x - 20)"
}
"""


DEFAULT_MAX_TOKENS = 16000

# USD per 1M tokens (input, output), Anthropic first-party list prices as of 2026-10.
# Thinking tokens are billed as output. Unknown models get token counts but no $ figure.
PRICING_PER_MTOK: Dict[str, Tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),   # prompts up to 100K tokens
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-fable-5-1": (10.00, 50.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def extract_usage(provider: str, raw_response: Any) -> Dict[str, int]:
    """Normalize token usage from a raw provider response to {input, output, thinking}."""
    if not isinstance(raw_response, dict):
        return {}
    if provider == "anthropic":
        u = raw_response.get("usage") or {}
        return {
            "input": int(u.get("input_tokens", 0)) + int(u.get("cache_creation_input_tokens", 0) or 0)
                     + int(u.get("cache_read_input_tokens", 0) or 0),
            "output": int(u.get("output_tokens", 0)),
            "thinking": int((u.get("output_tokens_details") or {}).get("thinking_tokens", 0)),
        }
    if provider == "openai":
        u = raw_response.get("usage") or {}
        return {"input": int(u.get("prompt_tokens", 0)), "output": int(u.get("completion_tokens", 0)), "thinking": 0}
    if provider == "gemini":
        u = raw_response.get("usageMetadata") or {}
        return {
            "input": int(u.get("promptTokenCount", 0)),
            "output": int(u.get("candidatesTokenCount", 0)) + int(u.get("thoughtsTokenCount", 0)),
            "thinking": int(u.get("thoughtsTokenCount", 0)),
        }
    return {}


def estimate_cost(model: str, usage: Dict[str, int]) -> Optional[float]:
    """USD cost for one call, or None if the model's price isn't known."""
    price = PRICING_PER_MTOK.get(model)
    if not price or not usage:
        return None
    return (usage.get("input", 0) * price[0] + usage.get("output", 0) * price[1]) / 1_000_000


class LLMNoAnswerError(RuntimeError):
    """The model returned no usable answer (e.g. it spent the whole output budget thinking)."""

    def __init__(self, message: str, raw_request: Any = None, raw_response: Any = None, thinking: Optional[str] = None):
        super().__init__(message)
        self.raw_request = raw_request
        self.raw_response = raw_response
        self.thinking = thinking


class LLMClient:
    """Handles communications with LLM providers (Gemini, OpenAI, Anthropic, or Mock)."""

    def __init__(self, provider: str = "gemini", model: Optional[str] = None, api_key: Optional[str] = None,
                 max_tokens: int = DEFAULT_MAX_TOKENS):
        self.provider = provider.lower()
        self.model = model or self._default_model()
        self.api_key = api_key or self._resolve_api_key()
        self.max_tokens = max_tokens  # caps thinking + answer together, i.e. the worst-case cost per attempt

    def _default_model(self) -> str:
        if self.provider == "gemini":
            return "gemini-2.5-flash"
        elif self.provider == "openai":
            return "gpt-4o"
        elif self.provider == "anthropic":
            return "claude-haiku-5-5"  # cheapest current Claude; use -m claude-sonnet-5-5 for showcase runs
        return "mock"

    def _resolve_api_key(self) -> Optional[str]:
        if self.provider == "gemini":
            return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        elif self.provider == "openai":
            return os.getenv("OPENAI_API_KEY")
        elif self.provider == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY")
        return None

    def query(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send the arena payload to the LLM and return parsed JSON output."""
        attempt_info = payload.get("attempt_info", {})
        curr_att = attempt_info.get("current_attempt", len(payload.get("history", [])) + 1)
        max_att = attempt_info.get("max_attempts", "N/A")
        rem_att = attempt_info.get("remaining_attempts", "N/A")

        user_message = (
            f"=== CURRENT ATTEMPT STATUS ===\n"
            f"Attempt {curr_att} of {max_att} ({rem_att} remaining)\n\n"
            f"Arena State & Attempt History:\n{json.dumps(payload, indent=2)}"
        )

        if self.provider == "gemini":
            return self._query_gemini(user_message)
        elif self.provider == "openai":
            return self._query_openai(user_message)
        elif self.provider == "anthropic":
            return self._query_anthropic(user_message)
        elif self.provider == "mock":
            return self._query_mock(payload)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _query_gemini(self, user_message: str) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is not set.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        
        req_body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"{SYSTEM_PROMPT}\n\n{user_message}"}]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": self.max_tokens,
                "temperature": 0.2,
                "responseMimeType": "application/json"
            }
        }

        data = json.dumps(req_body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

        try:
            with urllib.request.urlopen(req) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                parsed = self._parse_json_response(text)
                parsed["_raw_request"] = req_body
                parsed["_raw_response"] = result
                return parsed
        except Exception as e:
            raise RuntimeError(f"Gemini API call failed: {str(e)}")

    def _query_openai(self, user_message: str) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")

        url = "https://api.openai.com/v1/chat/completions"
        req_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": self.max_tokens,
            "temperature": 0.2
        }

        data = json.dumps(req_body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        )

        try:
            with urllib.request.urlopen(req) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                text = result["choices"][0]["message"]["content"]
                parsed = self._parse_json_response(text)
                parsed["_raw_request"] = req_body
                parsed["_raw_response"] = result
                return parsed
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {str(e)}")

    def _query_anthropic(self, user_message: str) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")

        url = "https://api.anthropic.com/v1/messages"
        req_body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            # Thinking runs by default on current models, but its text is omitted unless
            # requested. "summarized" returns a readable summary (the raw chain is never exposed).
            "thinking": {"type": "adaptive", "display": "summarized"},
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_message}]
        }

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        workspace_id = os.getenv("ANTHROPIC_WORKSPACE_ID")
        if workspace_id:
            headers["anthropic-workspace-id"] = workspace_id

        data = json.dumps(req_body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers)

        try:
            with urllib.request.urlopen(req) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                
                # Extract native thinking block(s) if present
                thinking_blocks = [
                    block.get("thinking", "") for block in result.get("content", [])
                    if block.get("type") == "thinking" and "thinking" in block
                ]
                thinking_text = "\n\n".join(thinking_blocks) if thinking_blocks else None

                text_blocks = [
                    block["text"] for block in result.get("content", [])
                    if block.get("type") == "text" and "text" in block
                ]
                text = "".join(text_blocks)
                stop_reason = result.get("stop_reason")

                if not text.strip():
                    reason = (
                        f"Ran out of output tokens ({self.max_tokens}) while thinking, before giving a formula"
                        if stop_reason == "max_tokens"
                        else f"Response contained no answer text (stop_reason={stop_reason})"
                    )
                    raise LLMNoAnswerError(reason, req_body, result, thinking_text)
                try:
                    parsed = self._parse_json_response(text)
                except ValueError:
                    if stop_reason == "max_tokens":
                        raise LLMNoAnswerError(
                            f"Ran out of output tokens ({self.max_tokens}) mid-answer; the JSON was cut off",
                            req_body, result, thinking_text,
                        )
                    raise
                if thinking_text:
                    parsed["_thinking"] = thinking_text
                parsed["_raw_request"] = req_body
                parsed["_raw_response"] = result
                return parsed
        except LLMNoAnswerError:
            raise
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            raise RuntimeError(f"Anthropic API call failed (HTTP {e.code}): {err_body}")
        except Exception as e:
            raise RuntimeError(f"Anthropic API call failed: {str(e)}")

    def _query_mock(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based fallback agent for testing when no API key is provided."""
        attempts = len(payload.get("history", [])) + 1
        shooter = payload["shooter"]
        target = payload["targets"][0]
        
        m = (target["y"] - shooter["y"]) / (target["x"] - shooter["x"])
        c = shooter["y"] - m * shooter["x"]

        if attempts == 1:
            res = {
                "reasoning": f"Attempt 1 (Mock): Direct baseline line from ({shooter['x']}, {shooter['y']}) to ({target['x']}, {target['y']}).",
                "strategy": "direct",
                "planned_waypoints": [{"x": shooter["x"], "y": shooter["y"]}, {"x": target["x"], "y": target["y"]}],
                "formula": f"{m:.4f} * x + {c:.4f}"
            }
        else:
            amp = 8.0 * (attempts - 1)
            freq = 0.15 + 0.05 * (attempts - 1)
            res = {
                "reasoning": f"Attempt {attempts} (Mock): Wave trajectory with amplitude {amp} and frequency {freq} to bypass obstacles.",
                "strategy": "trigonometric_wave",
                "planned_waypoints": [{"x": shooter["x"], "y": shooter["y"]}, {"x": 0.0, "y": amp}, {"x": target["x"], "y": target["y"]}],
                "formula": f"{amp:.2f} * sin({freq:.3f} * (x - {shooter['x']})) + ({m:.4f} * x + {c:.4f})"
            }
        res["_raw_request"] = {"provider": "mock", "payload": payload}
        res["_raw_response"] = {"provider": "mock", "output": {k: v for k, v in res.items() if not k.startswith("_")}}
        return res

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        
        # 1. Try direct json.loads first
        try:
            return json.loads(cleaned)
        except Exception:
            pass

        # 2. Try extracting content inside markdown code fences
        if "```" in cleaned:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except Exception:
                    pass

        # 3. Fallback: Find outermost '{' and '}' substring
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = cleaned[start_idx : end_idx + 1]
            try:
                return json.loads(json_str)
            except Exception as e:
                raise ValueError(f"Failed to parse JSON response from LLM: {str(e)}\nRaw output:\n{text}")

        raise ValueError(f"Failed to parse JSON response from LLM: Expecting JSON object '{{...}}'\nRaw output:\n{text}")


class SingleTestAgentRunner:
    """Executes a single benchmark scenario test run using an LLM agent."""

    def __init__(
        self,
        scenario_path: str,
        max_attempts: int = 5,
        provider: str = "gemini",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        output_dir: str = "outputs",
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.scenario_path = scenario_path
        self.max_attempts = max_attempts
        self.runner = BenchmarkRunner(scenario_path, output_dir=output_dir)
        self.llm_client = LLMClient(provider=provider, model=model, api_key=api_key, max_tokens=max_tokens)
        self.cost_log: list = []

    def _record_cost(self, attempt: int, raw_resp: Any) -> None:
        """Print this attempt's tokens and $ cost, and keep it for the run summary."""
        usage = extract_usage(self.llm_client.provider, raw_resp)
        if not usage:
            return
        cost = estimate_cost(self.llm_client.model, usage)
        self.cost_log.append({"attempt": attempt, **usage, "cost_usd": cost})
        cost_str = f"${cost:.4f}" if cost is not None else "price unknown"
        print(f"  Tokens:    in {usage['input']:,} | out {usage['output']:,} "
              f"(thinking {usage['thinking']:,}) | cost {cost_str}")

    def _finish(self, passed: bool, attempts_used: int) -> Tuple[bool, str]:
        """Print the run's total cost and write cost_summary.json into the run folder."""
        folder = self.runner.logger.test_folder
        known = [c["cost_usd"] for c in self.cost_log if c["cost_usd"] is not None]
        total = sum(known) if known else None
        summary = {
            "provider": self.llm_client.provider,
            "model": self.llm_client.model,
            "max_tokens": self.llm_client.max_tokens,
            "passed": passed,
            "attempts_used": attempts_used,
            "total_input_tokens": sum(c["input"] for c in self.cost_log),
            "total_output_tokens": sum(c["output"] for c in self.cost_log),
            "total_thinking_tokens": sum(c["thinking"] for c in self.cost_log),
            "total_cost_usd": round(total, 6) if total is not None else None,
            "pricing_per_mtok": PRICING_PER_MTOK.get(self.llm_client.model),
            "attempts": self.cost_log,
        }
        with open(os.path.join(folder, "cost_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 75)
        if passed:
            print(f"TEST PASSED in {attempts_used} attempt(s)!")
        else:
            print(f"TEST FAILED: Target was not hit within {self.max_attempts} attempts.")
        if self.cost_log:
            total_str = f"${total:.4f}" if total is not None else "price unknown"
            print(f"Run cost:      {total_str}  ({summary['total_output_tokens']:,} output tokens, "
                  f"{summary['total_thinking_tokens']:,} of them thinking)")
        print(f"Output folder: {folder}")
        print("\nTo view the trajectory playback in the 2D visualizer, run:")
        print("  python main.py")
        print("=" * 75)
        return passed, folder

    def _dump_attempt_files(self, attempt: int, raw_req: Any, raw_resp: Any, thinking: Optional[str],
                            llm_output: Dict[str, Any], no_answer: Optional[str] = None) -> None:
        """Write this attempt's raw API request/response and thinking summary into the run folder."""
        folder = self.runner.logger.test_folder
        if raw_req:
            with open(os.path.join(folder, f"attempt_{attempt}_raw_request.json"), "w", encoding="utf-8") as f:
                json.dump(raw_req, f, indent=2)
        if raw_resp:
            with open(os.path.join(folder, f"attempt_{attempt}_raw_response.json"), "w", encoding="utf-8") as f:
                json.dump(raw_resp, f, indent=2)

        with open(os.path.join(folder, f"attempt_{attempt}_thinking.txt"), "w", encoding="utf-8") as f:
            f.write(f"=== ATTEMPT {attempt} LLM THINKING DUMP ===\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
            if thinking:
                f.write("--- NATIVE LLM THINKING BLOCK ---\n")
                f.write(thinking)
                f.write("\n\n")
            if no_answer:
                f.write(f"--- NO ANSWER ---\n{no_answer}\n")
                return
            f.write("--- STRUCTURED REASONING & PLAN ---\n")
            f.write(f"Reasoning: {llm_output.get('reasoning')}\n")
            f.write(f"Strategy:  {llm_output.get('strategy')}\n")
            f.write(f"Formula:   y = {llm_output.get('formula')}\n")
            f.write(f"Waypoints: {json.dumps(llm_output.get('planned_waypoints'))}\n")

    def run(self) -> Tuple[bool, str]:
        print("=" * 75)
        print(f"GraphWars-LLM: Running Single Test Agent")
        print(f"Scenario:     {self.runner.scenario_name}")
        print(f"Provider:     {self.llm_client.provider} (Model: {self.llm_client.model})")
        print(f"Max Attempts: {self.max_attempts}")
        print(f"Max Tokens:   {self.llm_client.max_tokens} per attempt (thinking + answer)")
        print("=" * 75)

        for attempt in range(1, self.max_attempts + 1):
            print(f"\n[Attempt {attempt}/{self.max_attempts}] Querying LLM Agent...")

            # 1. Get LLM input payload (scenario + history)
            payload = self.runner.get_current_llm_payload()
            payload["attempt_info"] = {
                "current_attempt": attempt,
                "max_attempts": self.max_attempts,
                "remaining_attempts": self.max_attempts - attempt + 1,
            }

            # 2. Call LLM
            try:
                llm_response = self.llm_client.query(payload)
            except LLMNoAnswerError as e:
                # Counts as a missed attempt: logged, shown to the model next time, run continues
                self._dump_attempt_files(attempt, e.raw_request, e.raw_response, e.thinking, {}, no_answer=str(e))
                self.runner.record_no_answer(str(e))
                print(f"  [FAILED] No answer: {e}")
                self._record_cost(attempt, e.raw_response)
                continue
            except Exception as e:
                print(f"  [ERROR] Error during LLM query: {e}")
                return self._finish(False, attempt - 1)

            self._dump_attempt_files(
                attempt, llm_response.get("_raw_request"), llm_response.get("_raw_response"),
                llm_response.get("_thinking"), llm_response,
            )
            self._record_cost(attempt, llm_response.get("_raw_response"))

            print(f"  Reasoning: {llm_response.get('reasoning', 'N/A')}")
            print(f"  Strategy:  {llm_response.get('strategy', 'N/A')}")
            print(f"  Formula:   y = {llm_response.get('formula', 'N/A')}")

            # 3. Simulate shot in engine
            sim_result = self.runner.process_llm_response(llm_response)

            # 4. Display result
            if sim_result.is_success:
                print(f"  [SUCCESS] Target Destroyed ({sim_result.hit_target_id})!")
                print(f"     Closest distance: {sim_result.closest_distance_to_target:.3f}")
                return self._finish(True, attempt)

            else:
                print(f"  [FAILED] Shot Outcome: {sim_result.hit_type.value.upper()}")
                if sim_result.hit_obstacle_name:
                    print(f"     Collided with: {sim_result.hit_obstacle_name}")
                if sim_result.hit_coordinate:
                    print(f"     Impact Point:  ({sim_result.hit_coordinate.x:.2f}, {sim_result.hit_coordinate.y:.2f})")
                print(f"     Closest dist to target: {sim_result.closest_distance_to_target:.3f}")

        return self._finish(False, self.max_attempts)


def resolve_scenario_path(scenario_arg: str) -> str:
    """Resolve shorthand scenario names or file paths."""
    if os.path.exists(scenario_arg):
        return scenario_arg

    scenarios_dir = os.path.join("experiments", "scenarios")
    possible_file = os.path.join(scenarios_dir, scenario_arg)
    if os.path.exists(possible_file):
        return possible_file

    possible_json = os.path.join(scenarios_dir, f"{scenario_arg}.json")
    if os.path.exists(possible_json):
        return possible_json

    raise FileNotFoundError(f"Could not find scenario file: {scenario_arg}")


def main():
    parser = argparse.ArgumentParser(description="Run a single LLM agent benchmark test on a GraphWars scenario.")
    parser.add_argument(
        "--scenario", "-s",
        default="scenario_5_random_gauntlet.json",
        help="Path or name of the scenario JSON (default: scenario_5_random_gauntlet.json)"
    )
    parser.add_argument(
        "--max-attempts", "-a",
        type=int,
        default=5,
        help="Maximum allowed trajectory attempts (default: 5)"
    )
    parser.add_argument(
        "--provider", "-p",
        choices=["gemini", "openai", "anthropic", "mock"],
        default="anthropic",
        help="LLM Provider (default: anthropic, model claude-haiku-5-5)"
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Specific model name (e.g. claude-haiku-5-5, claude-sonnet-5-5, gemini-2.5-flash, gpt-4o)"
    )
    parser.add_argument(
        "--max-tokens", "-t",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Output token cap per attempt, thinking included (default: {DEFAULT_MAX_TOKENS}). "
             "Hitting it counts as a missed attempt.",
    )
    parser.add_argument(
        "--api-key", "-k",
        default=None,
        help="API Key (optional, defaults to environment variable GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY)"
    )

    args = parser.parse_args()

    try:
        scenario_path = resolve_scenario_path(args.scenario)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    runner = SingleTestAgentRunner(
        scenario_path=scenario_path,
        max_attempts=args.max_attempts,
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
    )
    success, _ = runner.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
