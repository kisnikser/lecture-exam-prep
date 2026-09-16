import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'

import { courseQuestionSets, findCourse } from '../lib/data'
import type { Coverage, DataIndex } from '../types/data'

const COVERAGE_LABEL: Record<Coverage, string> = {
  full: 'полный',
  partial: 'частичный',
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
            {set.kind === 'general' ? 'Общий список' : 'Список курса'} ({set.questions.length})
          </button>
        ))}
      </nav>
      {current && (
        <ol className="question-list">
          {current.questions.map((question) => {
            const coverage = coverageOf(question.id)
            return (
              <li key={question.id}>
                <Link to={`/course/${course.slug}/q/${question.id}`}>{question.text}</Link>
                <span className={`badge badge-${coverage ?? 'none'}`}>
                  {coverage ? COVERAGE_LABEL[coverage] : 'ответа нет'}
                </span>
              </li>
            )
          })}
        </ol>
      )}
    </section>
  )
}
