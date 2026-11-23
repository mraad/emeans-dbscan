# /// script
# requires-python = ">=3.12"
# dependencies = ["markdown"]
# ///
"""Build blog-post.html (standalone) and blogger-body.html from blog-post.md, images embedded."""
import base64
import pathlib
import re

import markdown

HERE = pathlib.Path(__file__).parent
text = (HERE / "blog-post.md").read_text()
m = re.match(r"\s*#\s+(.+?)\n", text)
assert m, "blog-post.md must start with a '# Title' line"
title, body = m.group(1).strip(), text[m.end():]
html = markdown.markdown(body, extensions=["fenced_code", "tables", "sane_lists"])


def inline(match):
    path = HERE / match.group(1)
    mime = "image/jpeg" if path.suffix == ".jpg" else "image/png"
    return f'src="data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"'


html = re.sub(r'src="(img/[^"]+)"', inline, html)
style = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:860px;margin:2em auto;padding:0 1em;line-height:1.55;color:#212529}
h1{font-size:1.7em;line-height:1.25} h2{margin-top:2em;border-bottom:1px solid #dee2e6;padding-bottom:.2em} h3{margin-top:1.5em}
table{border-collapse:collapse;margin:1em 0;font-size:.92em} th,td{border:1px solid #ced4da;padding:.35em .6em;text-align:left;vertical-align:top} th{background:#f1f3f5}
pre{background:#f8f9fa;border:1px solid #e9ecef;padding:.8em;overflow-x:auto;font-size:.88em} code{font-family:SFMono-Regular,Menlo,Consolas,monospace}
img{display:block;margin:1em auto}
"""
# Blogger: scope every rule to the post, so the blog theme keeps its own body and headings.
scoped = re.sub(r"(^|})\s*([^{}@]+){", lambda m: m.group(1) + ",".join(".em-post " + s.strip() for s in m.group(2).split(",")) + "{", style)
scoped = scoped.replace(".em-post body", ".em-post")
(HERE / "blogger-body.html").write_text(f'<style>{scoped}</style>\n<div class="em-post">\n{html}\n</div>\n')
(HERE / "blog-post.html").write_text(
    f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
    f"<title>{title}</title>\n<style>{style}</style>\n</head>\n<body>\n<article>\n<h1>{title}</h1>\n{html}\n</article>\n</body>\n</html>\n")
print(title)
