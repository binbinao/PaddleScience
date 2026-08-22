# Euler Beam 入门幻灯片

本目录中的 **`euler_beam_intro.md`** 为 [Marp](https://marp.app/) 格式的演示文稿，面向从零接触本案例的用户，内容与 `examples/euler_beam` 下脚本、配置及根目录 `README.md` 一致。

## 预览与导出

1. **VS Code / Cursor**  
   安装扩展 **Marp for VS Code**，打开 `euler_beam_intro.md`，使用预览与 **Export Slide Deck** 导出 HTML、PDF 或 PPTX。

2. **命令行（需 Node.js）**

   ```bash
   cd examples/euler_beam/slides
   npx @marp-team/marp-cli euler_beam_intro.md --html euler_beam_intro.html
   npx @marp-team/marp-cli euler_beam_intro.md --pdf euler_beam_intro.pdf
   ```

   导出 **PPTX** 时，若已安装 **LibreOffice**，可尝试：

   ```bash
   npx @marp-team/marp-cli euler_beam_intro.md --pptx euler_beam_intro.pptx
   ```

未安装 Marp 时，也可直接阅读 `euler_beam_intro.md` 源码中的分节（以 `---` 分隔幻灯片）。
