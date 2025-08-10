# MCP Servers 安装与配置指南

## 安装状态

✅ **已完成安装和配置的 MCP Servers:**

1. **Filesystem Server** - 文件系统操作
   - 路径: `mcp_servers/src/filesystem/dist/index.js`
   - 功能: 读写文件、目录操作、文件搜索
   - 状态: ✅ 正常运行

2. **Memory Server** - 知识图谱记忆
   - 路径: `mcp_servers/src/memory/dist/index.js`
   - 功能: 存储和检索对话记忆
   - 状态: ✅ 正常运行

3. **Everything Server** - 综合工具
   - 路径: `mcp_servers/src/everything/dist/index.js`
   - 功能: 多种实用工具集合
   - 状态: ✅ 已构建

4. **Fetch Server** - 网络请求
   - 路径: `mcp_servers/src/fetch/dist/index.js`
   - 功能: HTTP 请求和网页抓取
   - 状态: ✅ 已构建

5. **Sequential Thinking Server** - 序列思考 (本地)
   - 路径: `mcp_servers/src/sequentialthinking/dist/index.js`
   - 功能: 结构化思考过程
   - 状态: ✅ 已构建

6. **Smithery Sequential Thinking Server** - 序列思考 (云端)
   - 服务商: Smithery AI
   - 功能: 增强版结构化思考过程
   - 状态: ✅ 已安装配置

## 配置文件位置

- **Claude Desktop 配置**: `%APPDATA%\\Claude\\claude_desktop_config.json`
- **VS Code 工作区配置**: `.vscode\\settings.json`
- **项目配置备份**: `claude_desktop_config_final.json`

## 使用方法

### 1. 启动 Claude Desktop
重启 Claude Desktop 应用程序，MCP servers 会自动加载。

### 2. 验证连接
在 Claude Desktop 中，你现在可以：

- **文件操作**: 要求读取、编写、搜索项目文件
- **记忆功能**: Claude 可以记住跨对话的信息
- **网络请求**: 获取网页内容或 API 数据
- **工具集成**: 使用各种实用工具

### 3. 示例命令

```bash
# 文件系统操作示例
"请读取 fusion_xfeat_sfd2/README.md 文件"
"在项目中搜索包含 'semantic' 的文件"
"创建一个新的配置文件"

# 记忆功能示例  
"记住这个项目是关于 XFeat 和 SFD2 融合的"
"之前我们讨论过什么？"

# 网络请求示例
"获取 https://github.com/verlab/accelerated_features 的 README"

# 序列化思考示例
"请用结构化思考方式分析这个问题"
"帮我逐步思考 MCP 配置的优化方案"
```

## 故障排除

### 检查服务器状态
运行测试脚本：
```bash
cd mcp_servers
node test_servers.mjs
```

### 查看 Claude Desktop 日志
- Windows: `%APPDATA%\\Claude\\logs\\`
- 查找连接错误或启动问题

### 常见问题
1. **路径错误**: 确保所有路径使用正确的反斜杠格式
2. **权限问题**: 确保 Node.js 有访问项目目录的权限
3. **依赖缺失**: 运行 `npm install` 确保依赖完整

## 配置详情

当前配置允许 MCP servers 访问整个项目目录 (`C:\\Users\\王敬尧\\Desktop\\idea`)，包括：
- fusion_xfeat_sfd2/ (主项目)
- accelerated_features/ (XFeat 代码)
- sfd2/ (SFD2 代码)
- 所有其他项目文件

## 安全注意事项

⚠️ **重要**: MCP servers 有访问指定目录的完整权限，请确保：
1. 只在受信任的环境中使用
2. 定期备份重要文件
3. 监控文件系统操作日志

## 下一步

MCP servers 现在已配置完成。你可以：
1. 重启 Claude Desktop 开始使用
2. 测试各种 MCP 功能
3. 根据需要调整配置
4. 继续 fusion_xfeat_sfd2 项目开发
