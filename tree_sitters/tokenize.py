import os
from typing import List
from tree_sitter import Language, Parser, Node

java_lib = os.path.join(os.path.dirname(__file__), "tree-sitter-java.so")
JAVA_LANGUAGE = Language(java_lib, 'java')

parser = Parser()
parser.set_language(JAVA_LANGUAGE)


def parse_ast(s: str):
    tree = parser.parse(bytes(s, "utf8"))
    return tree


def tokenize(s: str):
    tree = parse_ast(s)
    tokens = []
    nodes_to_expand: List[Node] = [tree.root_node]
    while nodes_to_expand:
        node = nodes_to_expand.pop(0)
        if not node.children and node.text:
            tokens.append(node.text.decode())
        nodes_to_expand = node.children + nodes_to_expand
    return tokens


if __name__ == "__main__":
    code = """@Override
    public final String escape(String s) {
        checkNotNull(s); // GWT specific check (do not optimize)
        String a = "Hello world";
        /**
         * I'm a block comment.
         */
        int a = 0;
        char c = 'c';
        float b = 0.1 + 1e10 + -.5 + .4e4;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if ((c < replacementsLength && replacements[c] != null)
                || c > safeMaxChar
                || c < safeMinChar) {
                    return escapeSlow(s, i);
            }
        }
        return s;
    }"""
    tokens = tokenize(code)
    print(tokens)
