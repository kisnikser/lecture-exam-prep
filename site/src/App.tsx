import { useEffect, useState } from 'react'
import { HashRouter, Link, Route, Routes } from 'react-router-dom'

import { loadIndex } from './lib/data'
import { CoursePage } from './pages/CoursePage'
import { CoursesPage } from './pages/CoursesPage'
import { QuestionPage } from './pages/QuestionPage'
import type { DataIndex } from './types/data'

export function App() {
  const [index, setIndex] = useState<DataIndex | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadIndex()
      .then(setIndex)
      .catch((cause: unknown) => setError(String(cause)))
  }, [])

  return (
    <HashRouter>
      <header className="site-header">
        <Link to="/">Подготовка к экзамену</Link>
      </header>
      <main>
        {error && (
          <p className="error">
            Не удалось загрузить данные: {error}. Собери их командой{' '}
            <code>uv run examprep export-index</code>.
          </p>
        )}
        {!index && !error && <p className="muted">Загрузка…</p>}
        {index && (
          <Routes>
            <Route path="/" element={<CoursesPage index={index} />} />
            <Route path="/course/:slug" element={<CoursePage index={index} />} />
            <Route path="/course/:slug/q/:questionId" element={<QuestionPage index={index} />} />
          </Routes>
        )}
      </main>
    </HashRouter>
  )
}
