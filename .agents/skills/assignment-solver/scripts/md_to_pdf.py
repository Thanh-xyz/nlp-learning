import os
import sys
import re
import argparse
import subprocess
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def parse_markdown(md_content):
    lines = md_content.split('\n')
    latex_body = []
    
    in_code_block = False
    in_math_block = False
    in_table = False
    table_lines = []
    
    def process_inline(text):
        math_parts = []
        def math_repl(m):
            math_parts.append(m.group(0))
            return f"QQMATHINLINE{len(math_parts)-1}ZZ"
        text = re.sub(r'(\$[^\$\n]+?\$)', math_repl, text)
        
        code_parts = []
        def code_repl(m):
            code_parts.append(m.group(1))
            return f"QQCODEINLINE{len(code_parts)-1}ZZ"
        text = re.sub(r'`([^`\n]+?)`', code_repl, text)
        
        for char, repl in [
            ('&', r'\&'),
            ('%', r'\%'),
            ('#', r'\#'),
            ('_', r'\_'),
        ]:
            text = text.replace(char, repl)
            
        text = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', text)
        text = re.sub(r'\*(.+?)\*', r'\\textit{\1}', text)
        
        for i, code in enumerate(code_parts):
            esc = code.replace('\\', '\\textbackslash{}').replace('{', '\\{').replace('}', '\\}').replace('^', '\\textasciicircum{}').replace('~', '\\textasciitilde{}')
            text = text.replace(f"QQCODEINLINE{i}ZZ", r'\texttt{' + esc + '}')
            
        for i, math in enumerate(math_parts):
            text = text.replace(f"QQMATHINLINE{i}ZZ", math)
            
        return text

    def flush_table(tbl):
        if not tbl:
            return ""
        rows = []
        for line in tbl:
            clean = line.strip().lstrip('\ufeff').lstrip('\xa0')
            if re.match(r'^\|?\s*:?-+:?\s*(\|?\s*:?-+:?\s*)+\|?$', clean):
                continue
            cols = [c.strip() for c in clean.strip('|').split('|')]
            rows.append(cols)
        if not rows:
            return ""
        col_count = max(len(r) for r in rows)
        col_spec = " ".join([f">{{\\raggedright\\arraybackslash}}X" for _ in range(col_count)])
        
        out = [
            "\\vspace{0.6em}",
            "\\noindent",
            f"\\begin{{tabularx}}{{\\linewidth}}{{{col_spec}}}",
            "\\toprule"
        ]
        header = rows[0]
        while len(header) < col_count:
            header.append("")
        out.append(" & ".join([f"\\textbf{{{process_inline(c)}}}" for c in header]) + " \\\\")
        out.append("\\midrule")
        for r in rows[1:]:
            while len(r) < col_count:
                r.append("")
            out.append(" & ".join([process_inline(c) for c in r]) + " \\\\")
        out.append("\\bottomrule")
        out.append("\\end{tabularx}")
        out.append("\\vspace{0.6em}\n")
        return "\n".join(out)

    doc_title = None

    i = 0
    while i < len(lines):
        raw_line = lines[i]
        line = raw_line.rstrip()
        stripped = line.strip().lstrip('\ufeff').lstrip('\xa0')
        
        if stripped.startswith('```'):
            if not in_code_block:
                in_code_block = True
                latex_body.append("\\begin{lstlisting}")
            else:
                in_code_block = False
                latex_body.append("\\end{lstlisting}")
            i += 1
            continue
            
        if in_code_block:
            latex_body.append(raw_line)
            i += 1
            continue

        if stripped == '$$' or (stripped.startswith('$$') and stripped.endswith('$$') and len(stripped) > 2):
            if stripped == '$$':
                if not in_math_block:
                    in_math_block = True
                    latex_body.append("\\[")
                else:
                    in_math_block = False
                    latex_body.append("\\]")
            else:
                math_content = stripped[2:-2].strip()
                latex_body.append(f"\\[\n{math_content}\n\\]")
            i += 1
            continue

        if in_math_block:
            latex_body.append(raw_line)
            i += 1
            continue

        no_math_line = re.sub(r'\$[^\$\n]+?\$', '', stripped)
        is_tbl = (stripped.startswith('|') and stripped.endswith('|')) or (no_math_line.count('|') >= 2 and not stripped.startswith('#'))
        if is_tbl:
            if not in_table:
                in_table = True
                table_lines = [stripped]
            else:
                table_lines.append(stripped)
            i += 1
            continue
        else:
            if in_table:
                in_table = False
                latex_body.append(flush_table(table_lines))
                table_lines = []

        if stripped.startswith('# '):
            h_text = stripped[2:].strip()
            if not doc_title:
                doc_title = h_text
            else:
                latex_body.append(f"\\section*{{{process_inline(h_text)}}}")
        elif stripped.startswith('## '):
            latex_body.append(f"\\subsection*{{{process_inline(stripped[3:].strip())}}}")
        elif stripped.startswith('### '):
            latex_body.append(f"\\subsubsection*{{{process_inline(stripped[4:].strip())}}}")
        elif stripped.startswith('#### '):
            latex_body.append(f"\\paragraph*{{{process_inline(stripped[5:].strip())}}}")
        elif re.match(r'^[-*]\s+', stripped):
            item_text = re.sub(r'^[-*]\s+', '', stripped)
            latex_body.append(f"\\begin{{itemize}}[leftmargin=1.5em,noitemsep,topsep=2pt]\n\\item {process_inline(item_text)}\n\\end{{itemize}}")
        elif re.match(r'^\d+\.\s+', stripped):
            item_text = re.sub(r'^\d+\.\s+', '', stripped)
            latex_body.append(f"\\begin{{enumerate}}[leftmargin=1.5em,noitemsep,topsep=2pt]\n\\item {process_inline(item_text)}\n\\end{{enumerate}}")
        elif stripped.startswith('>'):
            quote_text = stripped.lstrip('>').strip()
            latex_body.append(f"\\begin{{quote}}\n{process_inline(quote_text)}\n\\end{{quote}}")
        elif not stripped:
            latex_body.append("\n")
        else:
            latex_body.append(process_inline(stripped) + "\n")
            
        i += 1

    if in_table:
        latex_body.append(flush_table(table_lines))

    return doc_title or "Báo Cáo Kỹ Thuật", "\n".join(latex_body)

def build_latex_document(title, body):
    return f"""\\documentclass[11pt,a4paper]{{article}}
\\usepackage{{fontspec}}
\\setmainfont{{Times New Roman}}
\\usepackage{{amsmath,amssymb,amsfonts}}
\\usepackage{{geometry}}
\\geometry{{a4paper, margin=0.8in}}
\\usepackage{{booktabs}}
\\usepackage{{tabularx}}
\\usepackage{{array}}
\\usepackage{{listings}}
\\usepackage{{xcolor}}
\\usepackage{{enumitem}}
\\usepackage{{fancyhdr}}
\\usepackage[colorlinks=true, linkcolor=blue, urlcolor=blue]{{hyperref}}

\\pagestyle{{fancy}}
\\fancyhf{{}}
\\rhead{{\\small\\textit{{{title}}}}}
\\lhead{{\\small Báo Cáo Kỹ Thuật}}
\\cfoot{{\\thepage}}

\\lstset{{
    basicstyle=\\ttfamily\\footnotesize,
    breaklines=true,
    frame=single,
    backgroundcolor=\\color{{black!3}},
    columns=fullflexible
}}

\\begin{{document}}
\\begin{{center}}
    {{\\LARGE\\textbf{{{title}}}}} \\\\[0.5em]
    {{\\normalsize Tài Liệu Phân Tích \\& Thực Nghiệm Khoa Học}} \\\\[0.3em]
    {{\\small\\textit{{\\today}}}}
\\end{{center}}
\\vspace{{0.8em}}
\\hrule
\\vspace{{1.2em}}

{body}

\\end{{document}}
"""

def convert_md_to_pdf(input_md_path, output_pdf_path=None, custom_title=None):
    in_path = Path(input_md_path).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")
        
    if output_pdf_path:
        out_path = Path(output_pdf_path).resolve()
    else:
        out_path = in_path.with_suffix('.pdf')

    with open(in_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    inferred_title, body = parse_markdown(md_text)
    title = custom_title or inferred_title

    tex_content = build_latex_document(title, body)
    tex_path = out_path.with_suffix('.tex')
    
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(tex_content)

    work_dir = tex_path.parent
    cmd = ["xelatex", "-interaction=nonstopmode", f"-output-directory={work_dir}", str(tex_path)]
    
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=str(work_dir))
    
    for ext in ['.aux', '.log', '.out']:
        aux_file = tex_path.with_suffix(ext)
        if aux_file.exists():
            try:
                aux_file.unlink()
            except Exception:
                pass

    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[SUCCESS] PDF generated: {out_path} ({out_path.stat().st_size} bytes)")
        return True
    else:
        print(f"[ERROR] Failed to compile PDF: {out_path}")
        print(result.stdout[-1500:])
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert Markdown with LaTeX formulas to PDF using XeLaTeX.")
    parser.add_argument("input_md", help="Path to input Markdown file")
    parser.add_argument("-o", "--output", help="Path to output PDF file (optional)")
    parser.add_argument("-t", "--title", help="Custom document title (optional)")
    args = parser.parse_args()

    success = convert_md_to_pdf(args.input_md, args.output, args.title)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
