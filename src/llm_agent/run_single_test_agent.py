import os
import sys
import json
import argparse
import urllib.request
import urllib.parse
from typing import Dict, Any, Optional, Tuple

from src.experiments.runner import BenchmarkRunner
from src.engine.backend.physics import TrajectoryResult


SYSTEM_PROMPT = """You are the Trajectory Planner for GraphWars-LLM in a continuous 2D Cartesian plane (x, y in R).
Your goal is to formulate an explicit mathematical function y = f(x) that starts at the shooter's coordinate, navigates around obstacles without colliding, and hits the target within its radius.

INPUT & ATTEMPT HISTORY ANALYSIS:
- You receive the arena geometry (shooter, target, bounds, obstacles) and an array of past attempts under 'history'.
- On Attempt 1, 'history' is empty [].
- On Attempt >1, 'history' contains your previous formulas and their exact simulation results (hit_type, hit_coordinate, hit_obstacle, closest_distance_to_target).
- CRITICAL: Carefully inspect the 'hit_coordinate' and 'hit_obstacle' in 'history' to adjust your curve's amplitude, frequency, or vertical shift to clear the obstacles you previously collided with!

CRITICAL MATHEMATICAL & SYNTAX RULES:
1. The formula MUST be an explicit function of 'x' (e.g. "0.05 * x**2 - 3").
2. Use standard Python math syntax:
   - Use '*' for multiplication (e.g., '2 * x', NOT '2x').
   - Use '**' or '^' for exponentiation (e.g., 'x**2').
   - Supported functions: sin, cos, tan, exp, log, sqrt, abs, pi, e.
3. Boundary conditions: f(x_shooter) must be close to y_shooter. f(x_target) must reach inside target radius.
4. Formula character length limit: strictly obey constraints.max_formula_characters.

OUTPUT CONTRACT:
You MUST respond with ONLY a valid JSON object matching this exact schema:
{
  "reasoning": "1-3 sentences analyzing obstacle clearance and trajectory shape based on arena state and previous attempt feedback",
  "strategy": "arc_over | s_curve | direct | low_tunnel | trigonometric_wave",
  "planned_waypoints": [
    {"x": -20.0, "y": 0.0},
    {"x": 0.0, "y": 8.0},
    {"x": 20.0, "y": 0.0}
  ],
  "formula": "-0.02 * (x + 20) * (x - 20)"
}
"""


class LLMClient:
    """Handles communications with LLM providers (Gemini, OpenAI, Anthropic, or Mock)."""

    def __init__(self, provider: str = "gemini", model: Optional[str] = None, api_key: Optional[str] = None):
        self.provider = provider.lower()
        self.model = model or self._default_model()
        self.api_key = api_key or self._resolve_api_key()

    def _default_model(self) -> str:
        if self.provider == "gemini":
            return "gemini-2.5-flash"
        elif self.provider == "openai":
            return "gpt-4o"
        elif self.provider == "anthropic":
            return "claude-3-5-sonnet-20241022"
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
        user_message = f"Arena State & Attempt History:\n{json.dumps(payload, indent=2)}"

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
                return self._parse_json_response(text)
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
                return self._parse_json_response(text)
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {str(e)}")

    def _query_anthropic(self, user_message: str) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")

        url = "https://api.anthropic.com/v1/messages"
        req_body = {
            "model": self.model,
            "max_tokens": 1024,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_message}]
        }

        data = json.dumps(req_body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
        )

        try:
            with urllib.request.urlopen(req) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                text = result["content"][0]["text"]
                return self._parse_json_response(text)
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
            return {
                "reasoning": f"Attempt 1 (Mock): Direct baseline line from ({shooter['x']}, {shooter['y']}) to ({target['x']}, {target['y']}).",
                "strategy": "direct",
                "planned_waypoints": [{"x": shooter["x"], "y": shooter["y"]}, {"x": target["x"], "y": target["y"]}],
                "formula": f"{m:.4f} * x + {c:.4f}"
            }
        else:
            amp = 8.0 * (attempts - 1)
            freq = 0.15 + 0.05 * (attempts - 1)
            return {
                "reasoning": f"Attempt {attempts} (Mock): Wave trajectory with amplitude {amp} and frequency {freq} to bypass obstacles.",
                "strategy": "trigonometric_wave",
                "planned_waypoints": [{"x": shooter["x"], "y": shooter["y"]}, {"x": 0.0, "y": amp}, {"x": target["x"], "y": target["y"]}],
                "formula": f"{amp:.2f} * sin({freq:.3f} * (x - {shooter['x']})) + ({m:.4f} * x + {c:.4f})"
            }

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response from LLM: {str(e)}\nRaw output: {text}")


class SingleTestAgentRunner:
    """Executes a single benchmark scenario test run using an LLM agent."""

    def __init__(
        self,
        scenario_path: str,
        max_attempts: int = 5,
        provider: str = "gemini",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        output_dir: str = "outputs"
    ):
        self.scenario_path = scenario_path
        self.max_attempts = max_attempts
        self.runner = BenchmarkRunner(scenario_path, output_dir=output_dir)
        self.llm_client = LLMClient(provider=provider, model=model, api_key=api_key)

    def run(self) -> Tuple[bool, str]:
        print("=" * 75)
        print(f"GraphWars-LLM: Running Single Test Agent")
        print(f"Scenario:     {self.runner.scenario_name}")
        print(f"Provider:     {self.llm_client.provider} (Model: {self.llm_client.model})")
        print(f"Max Attempts: {self.max_attempts}")
        print("=" * 75)

        for attempt in range(1, self.max_attempts + 1):
            print(f"\n[Attempt {attempt}/{self.max_attempts}] Querying LLM Agent...")

            # 1. Get LLM input payload (scenario + history)
            payload = self.runner.get_current_llm_payload()

            # 2. Call LLM
            try:
                llm_response = self.llm_client.query(payload)
            except Exception as e:
                print(f"  [ERROR] Error during LLM query: {e}")
                break

            print(f"  Reasoning: {llm_response.get('reasoning', 'N/A')}")
            print(f"  Strategy:  {llm_response.get('strategy', 'N/A')}")
            print(f"  Formula:   y = {llm_response.get('formula', 'N/A')}")

            # 3. Simulate shot in engine
            sim_result = self.runner.process_llm_response(llm_response)

            # 4. Display result
            if sim_result.is_success:
                print(f"  [SUCCESS] Target Destroyed ({sim_result.hit_target_id})!")
                print(f"     Closest distance: {sim_result.closest_distance_to_target:.3f}")
                print("\n" + "=" * 75)
                print(f"TEST PASSED in {attempt} attempt(s)!")
                print(f"Output folder: {self.runner.logger.test_folder}")
                print("\nTo view the trajectory playback in the 2D visualizer, run:")
                print("  python main.py")
                print("=" * 75)
                return True, self.runner.logger.test_folder

            else:
                print(f"  [FAILED] Shot Outcome: {sim_result.hit_type.value.upper()}")
                if sim_result.hit_obstacle_name:
                    print(f"     Collided with: {sim_result.hit_obstacle_name}")
                if sim_result.hit_coordinate:
                    print(f"     Impact Point:  ({sim_result.hit_coordinate.x:.2f}, {sim_result.hit_coordinate.y:.2f})")
                print(f"     Closest dist to target: {sim_result.closest_distance_to_target:.3f}")

        print("\n" + "=" * 75)
        print(f"TEST FAILED: Target was not hit within {self.max_attempts} attempts.")
        print(f"Output folder: {self.runner.logger.test_folder}")
        print("\nTo view the trajectory attempts in the 2D visualizer, run:")
        print("  python main.py")
        print("=" * 75)
        return False, self.runner.logger.test_folder


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
        default="gemini",
        help="LLM Provider (default: gemini)"
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Specific model name (e.g. gemini-2.5-flash, gpt-4o, claude-3-5-sonnet-20241022)"
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
        api_key=args.api_key
    )
    success, _ = runner.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
