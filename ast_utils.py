from tree_sitter import Language, Parser

Language.build_library(
    # Store the library in the `build` directory
    'build/tree_sitter.so',

    # Include one or more languages
    [
        'tree-sitter-python',
        'tree-sitter-java',
        'tree-sitter-cpp',
        'tree-sitter-c-sharp'
    ]
)

TS_LANG = {
    "python": Language('build/tree_sitter.so', 'python'),
    "java": Language('build/tree_sitter.so', 'java'),
    "cpp": Language('build/tree_sitter.so', 'cpp'),
    "csharp": Language('build/tree_sitter.so', 'c_sharp')
}


class ASTVisitor:

    def __init__(self, with_ndtypes=False, print_debug_outputs=False):
        self.with_ndtypes = with_ndtypes
        self.print_debug_outputs = print_debug_outputs
        self.stack = []
        self.ndtypes = []

    def enter(self, node) -> bool:
        return True

    def leave(self, node):
        pass

    def enter_leaf(self, node):
        pass

    def print_stack(self, node):
        depth = len(self.stack)
        print(" " * depth * 2 + node.type)

    def on_enter(self, node) -> bool:
        if self.print_debug_outputs:
            self.print_stack(node)
        if self.with_ndtypes:
            self.ndtypes.append((node.start_byte, True, node.type))
        enter_fn = getattr(self, "enter_%s" % node.type, self.enter)
        r = enter_fn(node)
        if node.child_count == 0:
            self.enter_leaf(node)
        self.stack.append(node.type)
        return r

    def on_leave(self, node):
        assert self.stack.pop() == node.type
        leave_fn = getattr(self, "leave_%s" % node.type, self.leave)
        r = leave_fn(node)
        # print("on leave ", node.type)
        if self.with_ndtypes:
            self.ndtypes.append((node.end_byte, False, node.type))
        return r

    def walk(self, root_node):
        if root_node is None:
            return

        cursor = root_node.walk()
        has_next = True

        while has_next:
            current_node = cursor.node

            # Step 1: Try to go to next child if we continue the subtree
            if self.on_enter(current_node):
                has_next = cursor.goto_first_child()
            else:
                has_next = False

            # Step 2: Try to go to next sibling
            if not has_next:
                self.on_leave(current_node)
                has_next = cursor.goto_next_sibling()

            # Step 3: Go up until sibling exists
            while not has_next and cursor.goto_parent():
                self.on_leave(cursor.node)  # We will never return to this specific parent
                has_next = cursor.goto_next_sibling()

    def __call__(self, root_node):
        return self.walk(root_node)


class ErrorCheckVisitor(ASTVisitor):
    def __init__(self, with_ndtypes=False):
        super().__init__(with_ndtypes)
        self.error_cnt = 0

    def enter_ERROR(self, node):
        if node.text.decode("utf-8") != ";":
            self.error_cnt += 1


def get_parser(lang):
    parser = Parser()
    parser.set_language(TS_LANG[lang])
    return parser
