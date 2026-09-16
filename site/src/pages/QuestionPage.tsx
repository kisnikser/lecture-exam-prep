import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

import { findQuestion, formatTimecode, loadAnswer, relatedQuestions, timecodeUrl } from '../lib/data'
import type { Answer, Coverage, DataIndex } from '../types/data'

const COVERAGE_LABEL: Record<Coverage, string> = {
  full: 'по лекциям',
  partial: 'частично',
  not_found: 'нет в лекциях',
}

export function QuestionPage({ index }: { index: DataIndex }) {
  const { slug = '', questionId = '' } = useParams()
  const found = findQuestion(index, questionId)
  const [answer, setAnswer] = useState<Answer | null>(null)
  const [missing, setMissing] = useState(false)

  useEffect(() => {
    setAnswer(null)
    setMissing(false)
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
      <Link className="breadcrumb" to={`/course/${slug}`}>
        ← к списку билетов
      </Link>

      <h1>
        {found.question.number}. {found.question.text}
      </h1>

      <div className="ticket-meta">
        {answer && (
          <span className={`badge badge-${answer.coverage}`}>{COVERAGE_LABEL[answer.coverage]}</span>
        )}
        {related.length > 0 && (
          <>
            <span className="muted">связанные:</span>
            {related.map((id) => (
              <Link key={id} to={`/course/${slug}/q/${id}`} className="chip">
                {id}
              </Link>
            ))}
          </>
        )}
      </div>

      {missing && <p className="muted">Ответ ещё не сгенерирован.</p>}

      {answer && (
        <>
          <div className="answer">
            <Markdown remarkPlugins={[remarkGfm]}>{answer.answer_md}</Markdown>
          </div>

          {answer.outside_lectures_md && (
            <aside className="outside">
              <span className="outside-label">Вне материалов лекций</span>
              <p className="muted">
                Лектор об этом не говорил. Ниже — общие знания по теме, чтобы закрыть пробел на
                экзамене.
              </p>
              <div className="answer">
                <Markdown remarkPlugins={[remarkGfm]}>{answer.outside_lectures_md}</Markdown>
              </div>
            </aside>
          )}

          {answer.readings.length > 0 && (
            <section>
              <h2>Литература, названная лектором</h2>
              <ul>
                {answer.readings.map((reading) => (
                  <li key={`${reading.video_id}-${reading.start}`}>
                    {reading.text}{' '}
                    <a
                      className="timecode"
                      href={timecodeUrl(reading.video_id, reading.start)}
                      target="_blank"
                      rel="noreferrer"
                    >
                      {formatTimecode(reading.start)}
                    </a>
                  </li>
                ))}
              </ul>
            </section>
          )}

          {answer.citations.length > 0 && (
            <section className="citations">
              <h2>Цитаты из лекций</h2>
              <ol>
                {answer.citations.map((citation) => (
                  <li key={citation.n}>
                    <div>
                      <a
                        className="timecode"
                        href={timecodeUrl(citation.video_id, citation.start)}
                        target="_blank"
                        rel="noreferrer"
                      >
                        {formatTimecode(citation.start)}
                      </a>
                      <q>{citation.quote}</q>
                    </div>
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
