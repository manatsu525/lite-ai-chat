# Lite AI Chat

一个面向低配 VPS 的轻量 AI 聊天网页，支持模型选择、登录认证、流式回答，以及真正独立于模型厂商的多轮外部搜索。

## 特性

- Groq Llama 3.3 70B、DeepSeek V4 Flash、DeepSeek V4 Pro 模型选择
- 外部 SearXNG 搜索层，不使用 DeepSeek 原生搜索
- 模型按需执行“分析结果 → 改写查询 → 再搜索”，最多五次外部检索
- 登录后可在网页填写 API Key、测试连接、读取并选择可用模型
- 保存最近 10 个服务器端对话窗口，支持切换、新建和删除
- 搜索后可抓取关键网页正文，优先引用官方来源
- 首次访问创建管理员账号，此后关闭公开注册
- 单进程 FastAPI，适合小内存 Debian/Ubuntu VPS
- 安装服务仅开放聊天端口；SearXNG 只监听 `127.0.0.1:8888`

## 系统要求

- Debian 11/12 或 Ubuntu 20.04+
- root 或 sudo 权限
- 建议至少 512 MB 内存；更低内存机器应配置 Swap
- 至少一个 Groq 或 DeepSeek API Key
- 服务器可以访问 Docker Hub、模型 API 和搜索引擎

## 一键安装

```bash
git clone https://github.com/manatsu525/lite-ai-chat.git
cd lite-ai-chat
sudo bash install.sh
```

安装过程中会隐藏输入 Groq、DeepSeek API Key，两者均可留空，但至少配置一个才能正常聊天。默认访问端口为 `8000`。

也可以非交互安装：

```bash
sudo GROQ_API_KEY='gsk_xxx' \
  DEEPSEEK_API_KEY='sk_xxx' \
  APP_PORT=8000 \
  bash install.sh
```

安装完成后访问：

```text
http://服务器IP:8000
```

第一次打开页面时创建管理员账号。

## 网页配置 API 与模型

登录后点击右上角的“API / 模型”：

1. 选择 Groq、DeepSeek 或其他 OpenAI 兼容接口。
2. 填写 API 地址和 API Key；已有密钥可留空继续使用。
3. 点击“测试 API 并读取模型”。
4. 从接口实际返回的模型列表中勾选需要显示的模型。
5. 保存后，聊天页的模型选择器会立即更新，无需重启服务。

密钥保存在服务器的 `data/providers.json`，权限为 `0600`。设置接口只返回密钥掩码，不会把完整密钥发送回浏览器。

取消勾选后保存可移除单个模型；点击“删除此 API 配置”可同时删除该提供商密钥和全部相关模型。

如果目标模型没有出现在 `/models` 返回结果中，可在“手动模型名”中每行填写一个模型。系统会直接向该模型发送最小聊天请求进行验证；即使提供商不支持 `/models`，手填模型验证成功后也可以保存。

### 自定义安装目录

```bash
sudo INSTALL_DIR=/opt/my-lite-ai-chat APP_PORT=9000 bash install.sh
```

升级时在仓库执行 `git pull`，然后重新运行安装脚本。已有 `.env` 和用户数据会保留。

## 配置

默认配置位于：

```text
/opt/lite-ai-chat/.env
```

修改后重启：

```bash
sudo systemctl restart lite-ai-chat
```

查看状态和日志：

```bash
systemctl status lite-ai-chat
journalctl -u lite-ai-chat -f
docker logs -f lite-ai-search
```

## 外部多轮搜索

所有可选模型都调用同一个 `web_search` 工具：

1. 模型生成第一条查询。
2. 本机 SearXNG 聚合外部搜索引擎并返回结果。
3. 模型判断信息是否充分；不足时使用年份、官网域名或 `site:` 改写查询。
4. 最多执行五次不同搜索；每次最多返回十条结果，并可抓取具体网页。
5. 当前模型基于工具结果组织最终答案和来源。

DeepSeek API 在本项目中只负责模型推理，不会调用 DeepSeek 原生搜索。

## 卸载

安全卸载会先把 `.env` 和用户数据库备份到 `/var/backups/`：

```bash
sudo bash uninstall.sh
```

彻底卸载且不保留备份：

```bash
sudo bash uninstall.sh --purge
```

卸载脚本不会删除系统 Docker、Python 或其他项目共用的软件包。

## 安全提示

- 不要提交 `.env`、数据库或真实 API Key。
- 建议使用防火墙限制聊天端口，或通过 Nginx/Caddy 配置 HTTPS。
- SearXNG 默认仅绑定本机，切勿无认证直接暴露到公网。
- 如果服务器已经被公网扫描，建议更换强管理员密码并启用 HTTPS。
