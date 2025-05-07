"""
Functions for drawing graphs using graphviz.

This module provides utilities for creating and formatting graphviz diagrams
of workflows, including HTML table generation and text escaping.
"""

from typing import Optional

def _html_escape(text: str) -> str:
    """Escape HTML special characters for use in HTML tables in graphviz.

    Replaces special characters with their HTML entity equivalents to ensure
    proper rendering in graphviz HTML-like labels.

    Args:
        text (str): The input text to escape.

    Returns:
        str: The escaped text with the following replacements:
        
            - ``&`` → ``&amp;``
            - ``<`` → ``&lt;``
            - ``>`` → ``&gt;``
            - ``Newline`` → ``<br align="left"/>``
    """
    text = str(text)
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br align="left"/>')

def create_node_label(title: str, id: Optional[str], duration: float, utilities: list, resource_deltas: dict, is_edge: bool = False) -> str:
    """Creates an HTML-formatted label string for use in graphviz diagrams, containing
    information about the node's title, duration, utilities, and resource changes.
    
    Args:
        title (str): Title of the node.
        duration (float): Duration of the node.
        utilities (list): List of utilities associated with the node.
        resource_deltas (dict): Dictionary of resource deltas associated with the node.
        is_edge (bool, optional): Whether the node represents an edge. Defaults to ``False``.
    
    Returns:
        str: An HTML-formatted string containing the node's label.
        
    Note:
        The returned string uses graphviz's HTML-like label syntax and includes:
        * A title section with optional edge formatting
        * Duration information
        * List of utilities
        * List of resource changes
    """
    edge_table_styles: str = 'cellborder="0" cellspacing="2" cellpadding="1" border="0"'
    node_table_styles: str = 'cellborder="0" cellspacing="2" cellpadding="1" border="1" style="rounded"'
    # Table content
    dur: str = _html_escape('+' + str(duration) if isinstance(duration, int) and duration > 0 else (str(duration) if duration else '--'))
    utils: str = ('<br align="left"/>' + '<br align="left"/>'.join([ str(idx + 1) + ') ' + _html_escape(x.value) for idx, x in enumerate(utilities) ])) if len(utilities) > 0 else '--' 
    resources: str = ('<br align="left"/>' + '<br align="left"/>'.join([ str(idx + 1) + ') ' + _html_escape(f"{'+' if v > 0 else ''}{round(v, 3)} {k}") for idx, (k, v) in enumerate(resource_deltas.items()) ])) if len(resource_deltas) > 0 else '--'
    id_str: str = (('<tr>'
                        '<td align="left">'
                            f'ID: {id}'
                        '</td>'
                    '</tr>') 
                if id is not None else '')
    return ('<<table ' + (edge_table_styles if is_edge else node_table_styles) + '>'
                '<tr><td height="2"></td></tr>'
                '<tr>'
                    '<td align="center" border="1" sides="B">'
                        f'{"<b><i>" if is_edge else "<b>"}'
                        f'{_html_escape(title)}'
                        f'{"</i></b>" if is_edge else "</b>"}'
                    '</td>'
                '</tr>'
                '<tr><td height="5"></td></tr>'
                f'{id_str}'
                '<tr>'
                    '<td align="left">'
                        f'Duration: {dur}'
                    '</td>'
                '</tr>'
                '<tr>'
                    '<td align="left">'
                        f'Utilities: {utils}'
                    '</td>'
                '</tr>'
                '<tr>'
                    '<td align="left">'
                        f'Resources: {resources}'
                    '</td>'
                '</tr>'
            '</table>>'
    )