# Web Interface Dockerfile
FROM node:18-alpine

WORKDIR /app

# 复制package文件
COPY gateway/web/package*.json ./

# 安装依赖
RUN npm ci --only=production

# 复制源代码
COPY gateway/web/ .

# 构建应用
RUN npm run build

# 暴露端口
EXPOSE 3000

# 启动命令
CMD ["npm", "start"]