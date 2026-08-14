FROM node:20-slim

WORKDIR /app

# Install pnpm
RUN npm install -g pnpm

# Copy repository dependency manifests
COPY pnpm-lock.yaml pnpm-workspace.yaml package.json ./
COPY packages/ ./packages/
COPY website/package.json ./website/

# Install dependencies across monorepo
RUN pnpm install

# Copy full source
COPY . .

# Expose Hugging Face default port 7860
EXPOSE 7860

# Launch the official Strudel REPL web server
CMD ["pnpm", "--filter", "@strudel/website", "dev", "--host", "0.0.0.0", "--port", "7860"]
