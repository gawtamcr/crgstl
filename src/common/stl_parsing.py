import re
from typing import Tuple, List
from common.stl_graph import STLNode, Predicate, And, Or, Eventually, Always

def parse_stl(formula: str) -> STLNode:
    """Parses an STL string into an STLNode tree."""
    formula = formula.strip()
    # Tokenize: Matches F[...], G[...], (, ), &, |, or words
    token_pattern = re.compile(r"([FG]\[[^\]]+\]|[()&|]|[a-zA-Z0-9_]+)")
    tokens = [t for t in token_pattern.findall(formula) if t.strip()]
    
    if not tokens:
        raise ValueError("Empty STL formula")

    def parse_expr(index: int) -> Tuple[STLNode, int]:
        # Parse terms separated by |
        lhs, index = parse_term(index)
        while index < len(tokens) and tokens[index] == '|':
            index += 1
            rhs, index = parse_term(index)
            lhs = Or(lhs, rhs)
        return lhs, index

    def parse_term(index: int) -> Tuple[STLNode, int]:
        # Parse factors separated by &
        lhs, index = parse_factor(index)
        while index < len(tokens) and tokens[index] == '&':
            index += 1
            rhs, index = parse_factor(index)
            lhs = And(lhs, rhs)
        return lhs, index

    def parse_factor(index: int) -> Tuple[STLNode, int]:
        if index >= len(tokens):
            raise ValueError("Unexpected end of formula")
            
        token = tokens[index]
        index += 1
        
        if token == '(':
            node, index = parse_expr(index)
            if index >= len(tokens) or tokens[index] != ')':
                raise ValueError("Missing closing parenthesis")
            return node, index + 1
            
        # Temporal Operators: F, G, F[0,10], G[0,5]
        if token == 'F' or token.startswith('F[') or token == 'G' or token.startswith('G['):
            op_type = Eventually if token.startswith('F') else Always
            t_start, t_end = 0.0, float('inf')
            
            if '[' in token:
                interval = token[token.find('[')+1 : token.find(']')]
                parts = interval.split(',')
                t_start = float(parts[0])
                if len(parts) > 1:
                    t_end = float(parts[1])
            
            child, index = parse_factor(index)
            return op_type(child, t_start, t_end), index
            
        # Predicate
        return Predicate(token), index

    root, _ = parse_expr(0)
    return root