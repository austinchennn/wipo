"""s1_parser — 解析招股书 S-1 / 招股说明书。

将招股书切分为三个模块:
  1. 产品信息 (product)
  2. 财务信息 (financial)
  3. 政策信息 (policy)

# TODO: 实现 PDF/HTML 招股书解析，建议使用 pdfplumber / unstructured。
#       输出应符合 ForumModel.raw_sections schema。
"""
