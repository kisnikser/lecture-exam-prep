import { cp } from 'node:fs/promises'
import { createReadStream, existsSync, statSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import react from '@vitejs/plugin-react'
import { defineConfig, type Plugin } from 'vite'

const DATA_DIR = fileURLToPath(new URL('../data', import.meta.url))

const CONTENT_TYPES: Record<string, string> = {
  '.json': 'application/json; charset=utf-8',
  '.jsonl': 'application/x-ndjson; charset=utf-8',
  '.txt': 'text/plain; charset=utf-8',
}

/**
 * Kept out of a published build.
 *
 * Audio and embeddings are simply too large. The transcripts and the chunks
 * cut from them are the lectures themselves, and a public site is a far more
 * visible place for someone else's course than a repository; the answers carry
 * their own quotes and link back to the original video, so nothing on the site
 * needs them, and neither does anything else on the site: extracts are debugging
 * artefacts of the extraction step. They stay available in dev, where a
 * transcript search can use them locally.
 */
const PRIVATE_TO_DEV = ['/audio', '/transcripts', '/extracts', 'chunks.jsonl', '.npy']

/** Serves `data/` at `/data/` in dev and copies it into the build output. */
function examprepData(): Plugin {
  return {
    name: 'examprep-data',
    configureServer(server) {
      server.middlewares.use('/data', (req, res, next) => {
        const relative = decodeURIComponent((req.url ?? '/').split('?')[0])
        const file = path.join(DATA_DIR, relative)
        if (!file.startsWith(DATA_DIR) || !existsSync(file) || !statSync(file).isFile()) {
          next()
          return
        }
        res.setHeader(
          'Content-Type',
          CONTENT_TYPES[path.extname(file)] ?? 'application/octet-stream',
        )
        // Пайплайн дописывает ответы прямо во время работы, и закэшированный
        // index.json показывает вчерашний список как сегодняшний.
        res.setHeader('Cache-Control', 'no-store')
        createReadStream(file).pipe(res)
      })
    },
    async closeBundle() {
      await cp(DATA_DIR, path.resolve('dist/data'), {
        recursive: true,
        filter: (source) => !PRIVATE_TO_DEV.some((part) => source.includes(part)),
      })
    },
  }
}

export default defineConfig({
  base: process.env.SITE_BASE ?? '/',
  plugins: [react(), examprepData()],
  test: {
    environment: 'node',
  },
})
