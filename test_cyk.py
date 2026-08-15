"""
Tests for cyk.py.

The integration tests reproduce exactly the outputs transcribed in commands.txt.
The unit tests exercise the module's internal functions.
"""

import io
import sys
import subprocess
import pytest
from pathlib import Path

REPO_DIR = Path(__file__).parent
PYTHON = sys.executable

# -- Import cyk without blocking on stdin -------------------------------------

class _FakeStdin:
    buffer = io.BytesIO(b"")

_orig_stdin = sys.stdin
sys.stdin = _FakeStdin()
sys.path.insert(0, str(REPO_DIR))
import cyk  # noqa: E402
sys.stdin = _orig_stdin


# -- Helpers ------------------------------------------------------------------

def run(*args, stdin_file: str | None = None, stdin_text: str | None = None) -> str:
    """Run cyk.py with the given arguments and return stdout."""
    cmd = [PYTHON, str(REPO_DIR / "cyk.py")] + list(args)
    if stdin_file:
        with open(REPO_DIR / stdin_file, "rb") as f:
            result = subprocess.run(cmd, stdin=f, capture_output=True, text=True, cwd=REPO_DIR)
    else:
        result = subprocess.run(
            cmd, input=stdin_text or "", capture_output=True, text=True, cwd=REPO_DIR
        )
    return result.stdout.replace("\r\n", "\n")


TREE_1 = "(S (A (B b) (A (B b) (A a))) (B b))"
TREE_2 = "(S (B b) (C (A (B b) (A a)) (B b)))"


# -- Integration tests (commands.txt) -----------------------------------------

class TestCommandsTxt:
    """
    Each method reproduces one command from commands.txt and checks the
    output matches exactly what is expected.
    """

    def test_g_g1_is_cnf(self):
        """$ python3 cyk.py -g g1.txt  ->  yes"""
        assert run("-g", "g1.txt") == "yes\n"

    def test_g_g2_not_cnf(self):
        """$ python3 cyk.py -g g2.txt  ->  no"""
        assert run("-g", "g2.txt") == "no\n"

    def test_p_in_g1(self):
        """$ cat in.txt | python3 cyk.py -p g1.txt  ->  yes\\nno"""
        assert run("-p", "g1.txt", stdin_file="in.txt") == "yes\nno\n"

    def test_t_in_g1(self):
        """$ cat in.txt | python3 cyk.py -t g1.txt  ->  two trees + blank line + no"""
        out = run("-t", "g1.txt", stdin_file="in.txt")
        lines = out.split("\n")

        assert TREE_1 in lines, f"Tree 1 not found in:\n{out}"
        assert TREE_2 in lines, f"Tree 2 not found in:\n{out}"

        no_idx = lines.index("no")
        assert lines[no_idx - 1] == "", "There must be a blank line right before 'no'"

    def test_t_tree_order_is_stable(self):
        """The assignment fixes the order the two trees are printed in; it is the
        insert(0, ...) in _fill_table that produces it."""
        lines = [l for l in run("-t", "g1.txt", stdin_file="in.txt").split("\n") if l.startswith("(")]
        assert lines == [TREE_1, TREE_2]

    def test_s_in_g1s(self):
        """$ cat in.txt | python3 cyk.py -s g1s.txt  ->  trees with probabilities + blank line + no"""
        out = run("-s", "g1s.txt", stdin_file="in.txt")
        lines = out.split("\n")

        tree_lines = [l for l in lines if l.startswith("(")]
        assert len(tree_lines) == 2, f"Expected 2 trees, got {len(tree_lines)}"

        trees: dict[str, float] = {}
        for l in tree_lines:
            tree_str, prob_str = l.rsplit(" ", 1)
            trees[tree_str] = float(prob_str)

        assert TREE_1 in trees, "Tree 1 not found"
        assert TREE_2 in trees, "Tree 2 not found"

        assert trees[TREE_1] == pytest.approx(0.02278, rel=1e-3)
        assert trees[TREE_2] == pytest.approx(0.02733, rel=1e-3)

        no_idx = lines.index("no")
        assert lines[no_idx - 1] == "", "There must be a blank line right before 'no'"


# -- Unit tests: is_cnf --------------------------------------------------------

class TestIsCNF:
    def test_g1_is_cnf(self):
        rules = ["SAB", "SBC", "ABA", "Aa", "BCC", "Bb", "CAB", "Ca"]
        assert cyk.is_cnf(rules) is True

    def test_g2_is_not_cnf(self):
        rules = ["SaSb", "Sab"]
        assert cyk.is_cnf(rules) is False

    def test_terminal_rule_is_valid(self):
        assert cyk.is_cnf(["Sa"]) is True

    def test_two_nonterminal_rule_is_valid(self):
        assert cyk.is_cnf(["SAB"]) is True

    def test_rule_too_long(self):
        assert cyk.is_cnf(["SABC"]) is False

    def test_uppercase_terminal_rejected(self):
        # A->B is not valid: the terminal must be lowercase
        assert cyk.is_cnf(["AB"]) is False

    def test_mixed_list(self):
        assert cyk.is_cnf(["SAB", "Aa", "SaSb"]) is False


# -- Unit tests: CFG -----------------------------------------------------------

class TestCFG:
    G1_RULES = ["SAB", "SBC", "ABA", "Aa", "BCC", "Bb", "CAB", "Ca"]

    def test_axiom_is_first_symbol(self):
        cfg = cyk.CFG(self.G1_RULES)
        assert cfg.axiom == "S"

    def test_body_ab_is_generated_by_s_and_c(self):
        cfg = cyk.CFG(self.G1_RULES)
        assert cfg.rules["AB"] == {"S", "C"}

    def test_terminal_body_a(self):
        cfg = cyk.CFG(self.G1_RULES)
        assert cfg.rules["a"] == {"A", "C"}

    def test_terminal_body_b(self):
        cfg = cyk.CFG(self.G1_RULES)
        # Bodies map to a SET of heads, not a list
        assert cfg.rules["b"] == {"B"}


# -- Unit tests: load_grammar --------------------------------------------------

class TestLoadGrammar:
    def test_load_g1_without_probabilities(self):
        rules, probs = cyk.load_grammar(str(REPO_DIR / "g1.txt"), False)
        assert "SAB" in rules
        assert probs == {}

    def test_load_g1s_with_probabilities(self):
        rules, probs = cyk.load_grammar(str(REPO_DIR / "g1s.txt"), True)
        assert "SAB" in rules
        assert probs["SAB"] == pytest.approx(0.25)
        assert probs["Bb"] == pytest.approx(0.9)

    def test_non_txt_extension(self):
        with pytest.raises(ValueError, match=r"\.txt"):
            cyk.load_grammar("grammar.csv", False)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            cyk.load_grammar("doesnotexist.txt", False)

    def test_duplicate_rule_is_kept_once(self):
        rules, _ = cyk.load_grammar(str(REPO_DIR / "dup.txt"), False)
        assert rules.count("SAB") == 1

    def test_probabilities_over_one_are_rejected(self):
        with pytest.raises(ValueError, match="more than one"):
            cyk.load_grammar(str(REPO_DIR / "bad_prob.txt"), True)

    def test_non_numeric_probability_is_rejected(self):
        with pytest.raises(ValueError, match="numeric"):
            cyk.load_grammar(str(REPO_DIR / "bad_fmt.txt"), True)

    def test_non_alphabetic_rule_is_rejected(self):
        with pytest.raises(ValueError, match="contiguous letters"):
            cyk.load_grammar(str(REPO_DIR / "noalpha.txt"), False)

    def test_probabilities_below_one_only_warn(self, capsys):
        # Falling short of 1 is a warning, not an error: the grammar still loads
        rules, probs = cyk.load_grammar(str(REPO_DIR / "low_prob.txt"), True)
        assert probs["SAB"] == pytest.approx(0.1)
        assert "do not reach one" in capsys.readouterr().err


# -- Unit tests: cyk_parse -----------------------------------------------------

class TestCYKParse:
    G1_RULES = ["SAB", "SBC", "ABA", "Aa", "BCC", "Bb", "CAB", "Ca"]

    def _cfg(self):
        return cyk.CFG(self.G1_RULES)

    def test_bbab_accepted(self, capsys):
        assert cyk.cyk_parse(self._cfg(), "bbab", False, {}) is True

    def test_bba_rejected(self, capsys):
        assert cyk.cyk_parse(self._cfg(), "bba", False, {}) is False

    def test_length_one_a_rejected(self, capsys):
        # S has no unit rules, so no string of length 1 belongs to the language
        assert cyk.cyk_parse(self._cfg(), "a", False, {}) is False

    def test_length_one_c_rejected(self, capsys):
        assert cyk.cyk_parse(self._cfg(), "c", False, {}) is False

    def test_symbol_outside_the_grammar_rejected(self, capsys):
        assert cyk.cyk_parse(self._cfg(), "zzzz", False, {}) is False

    def test_t_prints_the_trees(self, capsys):
        cyk.cyk_parse(self._cfg(), "bbab", pwt=True, probabilities={})
        captured = capsys.readouterr().out
        assert TREE_1 in captured
        assert TREE_2 in captured

    def test_ambiguous_string_yields_two_trees(self, capsys):
        """'bbab' is genuinely ambiguous under g1: the parser must enumerate both
        derivations, not just the first one it finds."""
        cyk.cyk_parse(self._cfg(), "bbab", pwt=True, probabilities={})
        lines = [l for l in capsys.readouterr().out.split("\n") if l.startswith("(")]
        assert len(lines) == 2
        assert len(set(lines)) == 2

    def test_s_prints_probabilities(self, capsys):
        _, probs = cyk.load_grammar(str(REPO_DIR / "g1s.txt"), True)
        cyk.cyk_parse(self._cfg(), "bbab", pwt=False, probabilities=probs)
        captured = capsys.readouterr().out
        lines = [l for l in captured.split("\n") if l.startswith("(")]
        assert len(lines) == 2
        prob_values = [float(l.rsplit(" ", 1)[1]) for l in lines]
        assert pytest.approx(0.02278, rel=1e-3) in prob_values
        assert pytest.approx(0.02733, rel=1e-3) in prob_values

    def test_probabilities_multiply_along_the_tree(self, capsys):
        """A tree's probability is the product of every rule used in it, so it must
        never exceed the probability of the rule at its root."""
        _, probs = cyk.load_grammar(str(REPO_DIR / "g1s.txt"), True)
        cyk.cyk_parse(self._cfg(), "bbab", pwt=False, probabilities=probs)
        lines = [l for l in capsys.readouterr().out.split("\n") if l.startswith("(")]
        for l in lines:
            assert float(l.rsplit(" ", 1)[1]) <= 0.75
