import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'

import { courseQuestionSets, findCourse } from '../lib/data'
import type { Coverage, DataIndex } from '../types/data'

/**
 * Покрытие решает, можно ли опираться на ответ, поэтому оно подписано словами,
 * а не только цветом: цвет читается быстро, но слово не оставляет догадок.
 */
const COVERAGE_LABEL: Record<Coverage, string> = {
  full: 'по лекциям',
  partial: 'частично',
  not_found: 'нет в лекциях',
}

export function CoursePage({ index }: { index: DataIndex }) {
  const { slug = '' } = useParams()
  const course = findCourse(index, slug)
  const sets = course ? courseQuestionSets(index, course) : []
  const [activeSet, setActiveSet] = useState(0)

  if (!course) {
    return <p className="error">Курс «{slug}» не найден.</p>
  }

  const current = sets[activeSet]
  const coverageOf = (questionId: string): Coverage | undefined =>
    course.answers.find((answer) => answer.question_id === questionId)?.coverage

  const ready = current
    ? current.questions.filter((question) => coverageOf(question.id)).length
    : 0

  return (
    <section>
      <h1>{course.title}</h1>

      <nav className="tabs">
        {sets.map((set, position) => (
          <button
            key={set.id}
            className={position === activeSet ? 'active' : ''}
            onClick={() => setActiveSet(position)}
          >
            {set.kind === 'general' ? 'Общий список' : 'Список курса'}{' '}
            <span className="question-number">{set.questions.length}</span>
          </button>
        ))}
      </nav>

      {current && (
        <>
          <p className="muted">
            Готово ответов: {ready} из {current.questions.length}
          </p>

          <ol className="question-list">
            {current.questions.map((question) => {
              const coverage = coverageOf(question.id)
              return (
                <li key={question.id}>
                  <Link to={`/course/${course.slug}/q/${question.id}`}>
                    <span className="question-number">{question.number}</span>
                    <span className="question-text">{question.text}</span>
                    <span className={`badge badge-${coverage ?? 'none'}`}>
                      {coverage ? COVERAGE_LABEL[coverage] : 'нет ответа'}
                    </span>
                  </Link>
                </li>
              )
            })}
          </ol>
        </>
      )}
    </section>
  )
}
