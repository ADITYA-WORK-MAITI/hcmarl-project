"""
Reusable framework for generating per-file code flowcharts.

Each diagram shows:
  - Boxes = functions / classes / code blocks within one file
  - Arrows between boxes = data/variables flowing between them
  - Dangling arrows INTO root boxes (imports from other files)
  - Dangling arrows OUT OF leaf boxes (exports consumed by other files)
  - Legend table (bottom-right): arrow_name | what_it_is | example
"""

import os
import graphviz

# Ensure Graphviz bin is on PATH (Windows)
os.environ["PATH"] = r"C:\Program Files\Graphviz\bin" + ";" + os.environ.get("PATH", "")

# ── Colour palette ──
HDR   = "#0F172A"   # file header (dark navy)
CLS   = "#1E3A5F"   # class header (steel blue)
FUNC  = "#2D4A22"   # standalone function (forest green)
CONST = "#6B3A1E"   # constants / module-level data (brown)
PROP  = "#4A2A5C"   # property / derived value (purple)
UTIL  = "#4A4A4A"   # utility / helper (grey)
EXT_PROD = "#3B82F6" # cross-file import (prod) — blue
EXT_TEST = "#10B981" # cross-file import (test) — green
EXT_IN   = "#DC2626" # incoming external data — red


def esc(text):
    """Escape text for Graphviz HTML labels."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class FlowchartBuilder:
    """Builder for a single-file code flowchart."""

    def __init__(self, file_path, purpose, equations="", lines="", imports_desc=""):
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.purpose = purpose
        self.equations = equations
        self.lines_desc = lines
        self.imports_desc = imports_desc

        self.g = graphviz.Digraph(
            self.file_name.replace(".py", "").replace(".yaml", ""),
            format="png",
        )
        self.g.attr(
            dpi="200", rankdir="TB", fontname="Consolas", bgcolor="white",
            nodesep="0.6", ranksep="0.8", margin="0.3", size="35,60",
        )
        self.g.attr("node", fontname="Consolas", fontsize="10", shape="none")
        self.g.attr("edge", fontname="Consolas", fontsize="8", color="#475569")

        # Track nodes for rank grouping
        self._rank_groups = {}  # rank_name -> [node_ids]
        self._legend_rows = []  # (arrow_name, what_it_is, example)

        # Create file header node
        rows = [("Purpose", purpose)]
        if equations:
            rows.append(("Equations", equations))
        if lines:
            rows.append(("Lines", lines))
        if imports_desc:
            rows.append(("Imports", imports_desc))
        self.make_node("file_hdr", self.file_path, HDR, rows)

    def make_node(self, node_id, title, color, rows):
        """Build an HTML-table Graphviz node.
        rows: list of (left_col, right_col) tuples.
        """
        hdr = (
            '<TR><TD COLSPAN="2" BGCOLOR="{}" ALIGN="LEFT">'
            '<FONT COLOR="white"><B>  {}  </B></FONT></TD></TR>'
        ).format(color, esc(title))
        body = ""
        for left, right in rows:
            r = right if right else " "
            body += (
                '<TR><TD ALIGN="LEFT"><FONT FACE="Consolas" POINT-SIZE="9">'
                '  {}  </FONT></TD>'
            ).format(esc(left))
            body += (
                '<TD ALIGN="LEFT"><FONT FACE="Consolas" POINT-SIZE="8" COLOR="#6B7280">'
                '  {}  </FONT></TD></TR>'
            ).format(esc(r))
        html = (
            '<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" '
            'CELLPADDING="3" COLOR="#CBD5E1">{}{}</TABLE>'
        ).format(hdr, body)
        self.g.node(node_id, "<" + html + ">")
        return node_id

    def add_external_node(self, node_id, label, is_test=False):
        """Add an external file node (plaintext, for dangling arrows)."""
        color = "#10B981" if is_test else "#94A3B8"
        self.g.node(
            node_id, label.replace("_", " "),
            shape="plaintext", fontname="Consolas", fontsize="8",
            fontcolor=color,
        )

    def edge(self, src, dst, label="", style="solid", color="#475569",
             constraint="true"):
        """Add a data-flow edge."""
        self.g.edge(
            src, dst,
            label=f"  {label}  " if label else "",
            style=style,
            color=color,
            constraint=constraint,
        )

    def dangling_in(self, target_node, label, source_file, color=EXT_IN):
        """Add a dangling arrow INTO a root node (from external file)."""
        phantom = f"_ext_in_{target_node}_{source_file.replace('.','_').replace('/','_')}"
        self.g.node(
            phantom,
            esc(source_file),
            shape="plaintext", fontname="Consolas", fontsize="7",
            fontcolor="#94A3B8",
        )
        self.g.edge(
            phantom, target_node,
            label=f"  {label}  ",
            style="dashed", color=color, arrowhead="normal",
        )

    def dangling_out(self, source_node, label, target_file,
                     is_test=False, color=None):
        """Add a dangling arrow OUT OF a leaf node (to external file)."""
        if color is None:
            color = EXT_TEST if is_test else EXT_PROD
        phantom = f"_ext_out_{source_node}_{target_file.replace('.','_').replace('/','_')}"
        self.g.node(
            phantom,
            esc(target_file),
            shape="plaintext", fontname="Consolas", fontsize="7",
            fontcolor="#10B981" if is_test else "#94A3B8",
        )
        self.g.edge(
            source_node, phantom,
            label=f"  {label}  ",
            style="dotted", color=color, dir="forward",
        )

    def add_legend_entry(self, arrow_name, what_it_is, example):
        """Add a row to the legend table."""
        self._legend_rows.append((arrow_name, what_it_is, example))

    def set_rank(self, rank_name, node_ids):
        """Group nodes at the same rank."""
        self._rank_groups[rank_name] = node_ids

    def build_legend(self):
        """Build the legend node from accumulated entries."""
        if not self._legend_rows:
            return
        rows_html = ""
        rows_html += (
            '<TR>'
            '<TD BGCOLOR="#334155"><FONT COLOR="white" POINT-SIZE="8"><B>Arrow</B></FONT></TD>'
            '<TD BGCOLOR="#334155"><FONT COLOR="white" POINT-SIZE="8"><B>What it is</B></FONT></TD>'
            '<TD BGCOLOR="#334155"><FONT COLOR="white" POINT-SIZE="8"><B>Example</B></FONT></TD>'
            '</TR>'
        )
        for name, desc, ex in self._legend_rows:
            rows_html += (
                '<TR>'
                '<TD ALIGN="LEFT"><FONT POINT-SIZE="7" COLOR="#475569">  {}  </FONT></TD>'
                '<TD ALIGN="LEFT"><FONT POINT-SIZE="7" COLOR="#475569">  {}  </FONT></TD>'
                '<TD ALIGN="LEFT"><FONT POINT-SIZE="7" COLOR="#6B7280">  {}  </FONT></TD>'
                '</TR>'
            ).format(esc(name), esc(desc), esc(ex))

        legend_html = (
            '<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" '
            'CELLPADDING="4" BGCOLOR="#F8FAFC" COLOR="#CBD5E1">'
            '<TR><TD COLSPAN="3" BGCOLOR="#334155">'
            '<FONT COLOR="white"><B>  DATA FLOW LEGEND  </B></FONT></TD></TR>'
            '{}</TABLE>'
        ).format(rows_html)
        self.g.node("legend", "<" + legend_html + ">")

    def render(self, output_dir="diagrams/code_flowcharts", stem_override=None):
        """Apply ranks, build legend, and render to PNG."""
        # Apply rank constraints
        for rank_name, node_ids in self._rank_groups.items():
            with self.g.subgraph() as s:
                s.attr(rank="same")
                for nid in node_ids:
                    s.node(nid)

        # Build legend
        self.build_legend()

        # Render
        if stem_override:
            stem = stem_override
        else:
            stem = self.file_name.replace(".py", "").replace(".ipynb", "").replace(".yaml", "").replace(".yml", "")
        out_path = os.path.join(output_dir, f"flowchart_{stem}")
        self.g.render(out_path, cleanup=True)
        print(f"OK: {out_path}.png")
        return f"{out_path}.png"
