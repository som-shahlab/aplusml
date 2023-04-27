
def _html_escape(text: str):
    """Escape HTML special characters: &, <, >, and " for use in HTML table in graphviz"""
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br align="left"/>')

def create_node_label(title: str, duration: float, utilities: list, resource_deltas: dict, is_edge: bool = False) -> str:
    """Create label for graphviz node"""
    # Styles
    edge_table_styles: str = 'cellborder="0" cellspacing="2" cellpadding="1" border="0"'
    node_table_styles: str = 'cellborder="0" cellspacing="2" cellpadding="1" border="1" style="rounded"'
    # Table content
    dur: str = _html_escape('+' + str(duration) if isinstance(duration, int) and duration > 0 else (str(duration) if duration else '--'))
    utils: str = ('<br align="left"/>' + '<br align="left"/>'.join([ str(idx + 1) + ') ' + _html_escape(x.value) for idx, x in enumerate(utilities) ])) if len(utilities) > 0 else '--' 
    resources: str = ('<br align="left"/>' + '<br align="left"/>'.join([ str(idx + 1) + ') ' + _html_escape(f"{'+' if v > 0 else ''}{round(v, 3)} {k}") for idx, (k, v) in enumerate(resource_deltas.items()) ])) if len(resource_deltas) > 0 else '--'
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