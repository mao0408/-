# macOS 话术助手

这是独立微信回复助手的 Tauri/macOS 外壳。回复引擎、OCR、训练记忆和发送逻辑复用上级目录的 Python 模块。

开发时先启动本地服务：

```bash
python ../mac_backend.py
```

再进入本目录安装 Tauri 依赖并启动开发窗口：

```bash
npm install
npx tauri dev
```

正式打包前，在 macOS 上构建 Python 后端：

```bash
python ../build_mac_backend.py
npx tauri build
```
