import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

import { findQuestion, formatTimecode, loadAnswer, relatedQuestions, timecodeUrl } from '../lib/data'
import type { Answer, DataIndex } from '../types/data'

export function QuestionPage({ index }: { index: DataIndex }) {
  const { slug = '', questionId = '' } = useParams()
  const found = findQuestion(index, questionId)
  const [answer, setAnswer] = useState<Answer | null>(null)
  const [missing, setMissing] = useState(false)
  const [revealed, setRevealed] = useState(false)

  useEffect(() => {
    setAnswer(null)
    setMissing(false)
    setRevealed(false)
    loadAnswer(slug, questionId)
      .then(setAnswer)
      .catch(() => setMissing(true))
  }, [slug, questionId])

  if (!found) {
    return <p className="error">Вопрос «{questionId}» не найден.</p>
  }

  const related = relatedQuestions(index, questionId)

  return (
    <article>
      <p className="muted">
        <Link to={`/course/${slug}`}>← к списку билетов</Link>
      </p>
      <h1>
        {found.question.number}. {found.question.text}
      </h1>

      {related.length > 0 && (
        <p className="muted">
          Связанные билеты:{' '}
          {related.map((id) => (
            <Link key={id} to={`/course/${slug}/q/${id}`} className="chip">
              {id}
            </Link>
          ))}
        </p>
      )}

      {missing && <p className="muted">Ответ ещё не сгенерирован.</p>}

      {answer && !revealed && (
        <button className="reveal" onClick={() => setRevealed(true)}>
          Показать ответ
        </button>
      )}

      {answer && revealed && (
        <>
          <Markdown remarkPlugins={[remarkGfm]}>{answer.answer_md}</Markdown>

          {answer.outside_lectures_md && (
            <section className="outside">
              <h2>Вне материалов лекций</h2>
              <Markdown remarkPlugins={[remarkGfm]}>{answer.outside_lectures_md}</Markdown>
            </section>
          )}

          {answer.readings.length > 0 && (
            <section>
              <h2>Литература (по лекциям)</h2>
              <ul>
                {answer.readings.map((reading) => (
                  <li key={`${reading.video_id}-${reading.start}`}>
                    {reading.text}{' '}
                    <a href={timecodeUrl(reading.video_id, reading.start)} target="_blank" rel="noreferrer">
                      {formatTimecode(reading.start)}
                    </a>
                  </li>
                ))}
              </ul>
            </section>
          )}

          {answer.citations.length > 0 && (
            <section className="citations">
              <h2>Цитаты</h2>
              <ol>
                {answer.citations.map((citation) => (
                  <li key={citation.n}>
                    <a href={timecodeUrl(citation.video_id, citation.start)} target="_blank" rel="noreferrer">
                      {formatTimecode(citation.start)}
                    </a>{' '}
                    <q>{citation.quote}</q>
                  </li>
                ))}
              </ol>
            </section>
          )}
        </>
      )}
    </article>
  )
}
