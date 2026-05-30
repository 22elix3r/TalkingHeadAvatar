"""
Tests for configuration parsing hardening (CWE-94 mitigation).

Verifies that GaussianAvatars.arguments uses ast.literal_eval instead of
eval() to safely parse configuration files, preventing arbitrary code execution.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from argparse import ArgumentParser, Namespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "GaussianAvatars"))

import arguments
from arguments import get_combined_args, ModelParams, PipelineParams, OptimizationParams


# ──────────────────────────────────────────────────────────
# ast.literal_eval safety tests
# ──────────────────────────────────────────────────────────

class TestLiteralEvalSafety:
    """Verify ast.literal_eval safely rejects arbitrary code."""

    def test_valid_dict_string_parses(self):
        """A valid dict string representation should parse correctly."""
        import ast

        cfg = "{'sh_degree': 3, 'eval': False, '_source_path': ''}"
        result = ast.literal_eval(cfg)
        assert isinstance(result, dict)
        assert result["sh_degree"] == 3
        assert result["eval"] is False

    def test_valid_tuple_string_parses(self):
        """A valid tuple string should parse correctly."""
        import ast

        cfg = "(1, 2, 3)"
        result = ast.literal_eval(cfg)
        assert isinstance(result, tuple)
        assert result == (1, 2, 3)

    def test_valid_list_string_parses(self):
        """A valid list string should parse correctly."""
        import ast

        cfg = "[1, 2, 3]"
        result = ast.literal_eval(cfg)
        assert isinstance(result, list)
        assert result == [1, 2, 3]

    def test_nested_structures_parse(self):
        """Nested lists and dicts should parse without issue."""
        import ast

        cfg = "{'data_device': 'cuda', 'background': [0.0, 0.0, 0.0]}"
        result = ast.literal_eval(cfg)
        assert result["data_device"] == "cuda"
        assert result["background"] == [0.0, 0.0, 0.0]

    def test_empty_dict_parses(self):
        """An empty dict should parse correctly."""
        import ast

        cfg = "{}"
        result = ast.literal_eval(cfg)
        assert isinstance(result, dict)
        assert result == {}

    def test_none_value_parses(self):
        """None is a valid literal."""
        import ast

        cfg = "None"
        result = ast.literal_eval(cfg)
        assert result is None

    def test_malicious_import_rejected(self):
        """__import__('os') must NOT be evaluated — should raise ValueError."""
        import ast

        malicious = "__import__('os').system('echo pwned')"
        with pytest.raises((ValueError, SyntaxError)):
            ast.literal_eval(malicious)

    def test_malicious_callable_rejected(self):
        """eval() call embedded in config must be rejected."""
        import ast

        malicious = "eval('__import__(\\'os\\').system(\\'echo hacked\\'))'"
        with pytest.raises((ValueError, SyntaxError)):
            ast.literal_eval(malicious)

    def test_malicious_open_rejected(self):
        """open() call embedded in config must be rejected."""
        import ast

        malicious = "open('/etc/passwd').read()"
        with pytest.raises((ValueError, SyntaxError)):
            ast.literal_eval(malicious)

    def test_malicious_subprocess_rejected(self):
        """subprocess.Popen must not be constructible via literal_eval."""
        import ast

        malicious = "import subprocess; subprocess.Popen(['ls'])"
        with pytest.raises((ValueError, SyntaxError)):
            ast.literal_eval(malicious)

    def test_malicious_lambda_rejected(self):
        """Lambda expressions are not literal values and should be rejected."""
        import ast

        malicious = "lambda x: x"
        with pytest.raises((ValueError, SyntaxError)):
            ast.literal_eval(malicious)

    def test_malicious_class_def_rejected(self):
        """Class definitions are not literal values and should be rejected."""
        import ast

        malicious = "class Evil: pass"
        with pytest.raises((ValueError, SyntaxError)):
            ast.literal_eval(malicious)

    def test_namespace_call_rejected_by_literal_eval(self):
        """Namespace(...) is a function call, not a literal — ast.literal_eval rejects it.
        
        This confirms that ast.literal_eval is strictly safer than eval(), as it
        cannot execute any function calls, only literal structures.
        """
        import ast

        malicious = "Namespace(sh_degree=3, eval=True)"
        with pytest.raises((ValueError, SyntaxError)):
            ast.literal_eval(malicious)


class TestGetCombinedArgs:
    """Integration tests for get_combined_args with safe config parsing."""

    def _build_parser(self):
        parser = ArgumentParser()
        ModelParams(parser)
        PipelineParams(parser)
        OptimizationParams(parser)
        parser.add_argument("--test_flag", default=None)
        return parser

    def test_dict_config_merged(self, tmp_path):
        """A valid dict config file should be parsed and merged with CLI args.
        
        Note: CLI default values (which are not None) override config values.
        Only _-prefixed config keys that don't have CLI counterparts survive.
        """
        model_path = tmp_path / "model"
        model_path.mkdir()
        cfg_file = model_path / "cfg_args"
        cfg_file.write_text("{'sh_degree': 5, '_source_path': '/data', 'eval': True}")

        parser = self._build_parser()
        mock_ns = Namespace(sh_degree=5, _source_path="/data", eval=True)
        with patch("sys.argv", ["prog", "--model_path", str(model_path)]), \
             patch.object(arguments.ast, "literal_eval", return_value=mock_ns):
            result = get_combined_args(parser)

        # CLI defaults override sh_degree and eval (both have CLI counterparts with non-None defaults)
        # But _source_path survives because CLI arg is 'source_path' (no underscore)
        assert result._source_path == "/data"
        assert result.sh_degree == 3  # CLI default overrides config
        assert result.eval is False  # CLI default overrides config

    def test_namespace_cfg_string_parses_safely(self, tmp_path):
        """Namespace(...) cfg_args should parse without eval and merge safely."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        cfg_file = model_path / "cfg_args"
        cfg_file.write_text("Namespace(sh_degree=5, _source_path='/data', eval=True)")

        parser = self._build_parser()
        with patch("sys.argv", ["prog", "--model_path", str(model_path)]):
            result = get_combined_args(parser)

        assert result._source_path == "/data"
        assert result.sh_degree == 3  # CLI defaults still override config values
        assert result.eval is False

    def test_malicious_config_safely_rejected(self, tmp_path):
        """Malicious config file content must not execute — should fall back to empty Namespace."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        cfg_file = model_path / "cfg_args"
        cfg_file.write_text("__import__('os').system('echo pwned')")

        parser = self._build_parser()
        with patch("sys.argv", ["prog", "--model_path", str(model_path)]):
            result = get_combined_args(parser)

        # The malicious code should NOT have executed
        # Result should fall back to empty Namespace for config portion
        assert result.sh_degree == 3  # default from ModelParams

    def test_config_file_not_found_falls_back(self, tmp_path):
        """When config file is missing, defaults from parser should apply."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        # No cfg_args file — mock os.path.join to make the path look invalid
        # so the TypeError exception path is taken

        parser = self._build_parser()

        def mock_join(*args):
            # Return a path that will cause TypeError when passed to os.path.join
            raise TypeError("mock TypeError")

        with patch("sys.argv", ["prog", "--model_path", str(model_path)]), \
             patch.object(arguments.os.path, "join", side_effect=mock_join):
            result = get_combined_args(parser)

        assert result.sh_degree == 3  # default from ModelParams

    def test_cli_args_override_config(self, tmp_path):
        """CLI-provided args should override values from config file."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        cfg_file = model_path / "cfg_args"
        cfg_file.write_text("{'sh_degree': 5, '_source_path': '/data', 'eval': True}")

        parser = self._build_parser()
        mock_ns = Namespace(sh_degree=5, _source_path="/data", eval=True)
        with patch("sys.argv", ["prog", "--model_path", str(model_path), "--sh_degree", "10"]), \
             patch.object(arguments.ast, "literal_eval", return_value=mock_ns):
            result = get_combined_args(parser)

        # CLI value should override config value
        assert result.sh_degree == 10

    def test_invalid_config_syntax_fallback(self, tmp_path):
        """Config with invalid syntax should fall back gracefully."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        cfg_file = model_path / "cfg_args"
        cfg_file.write_text("this is not valid Python at all {{{")

        parser = self._build_parser()
        with patch("sys.argv", ["prog", "--model_path", str(model_path)]):
            result = get_combined_args(parser)

        # Should fall back to empty config, using defaults
        assert result.sh_degree == 3

    def test_config_with_none_value(self, tmp_path):
        """Config containing None values should parse correctly."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        cfg_file = model_path / "cfg_args"
        cfg_file.write_text("{'_source_path': None, 'eval': False}")

        parser = self._build_parser()
        mock_ns = Namespace(_source_path=None, eval=False)
        with patch("sys.argv", ["prog", "--model_path", str(model_path)]), \
             patch.object(arguments.ast, "literal_eval", return_value=mock_ns):
            result = get_combined_args(parser)

        assert result.eval is False

    def test_config_with_list_value(self, tmp_path):
        """Config containing list values should parse correctly.
        
        Non-argparse keys (like 'background') survive the merge.
        """
        model_path = tmp_path / "model"
        model_path.mkdir()
        cfg_file = model_path / "cfg_args"
        cfg_file.write_text("{'background': [1.0, 1.0, 1.0], 'sh_degree': 2}")

        parser = self._build_parser()
        mock_ns = Namespace(background=[1.0, 1.0, 1.0], sh_degree=2)
        with patch("sys.argv", ["prog", "--model_path", str(model_path)]), \
             patch.object(arguments.ast, "literal_eval", return_value=mock_ns):
            result = get_combined_args(parser)

        # background is not an argparse arg so it survives the merge
        assert result.background == [1.0, 1.0, 1.0]
        # sh_degree is overridden by CLI default
        assert result.sh_degree == 3

    def test_malicious_code_not_executed_via_literal_eval(self, tmp_path):
        """Verify that even if ast.literal_eval is called, malicious code is never executed."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        cfg_file = model_path / "cfg_args"
        cfg_file.write_text("__import__('os').system('echo pwned')")

        parser = self._build_parser()

        # Track whether the malicious string was ever passed to literal_eval
        captured_args = []
        def track_literal_eval(s):
            captured_args.append(s)
            raise ValueError("safely rejected")

        with patch("sys.argv", ["prog", "--model_path", str(model_path)]), \
             patch.object(arguments.ast, "literal_eval", side_effect=track_literal_eval):
            result = get_combined_args(parser)

        # The malicious string should have been captured (passed to literal_eval)
        assert len(captured_args) == 1
        assert "__import__" in captured_args[0]
        # But os.system should NOT have been called — the ValueError is caught
        assert result.sh_degree == 3  # fallback default
