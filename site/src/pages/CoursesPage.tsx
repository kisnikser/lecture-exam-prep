import { Link } from 'react-router-dom'

import type { DataIndex } from '../types/data'

export function CoursesPage({ index }: { index: DataIndex }) {
  return (
    <section>
      <h1>Курсы</h1>
      <ul className="card-list">
        {index.courses.map((course) => (
          <li key={course.slug}>
            <Link to={`/course/${course.slug}`}>
              <strong>{course.title}</strong>
              <p className="muted">
                {course.videos} лекций · {course.answers.length} готовых ответов
              </p>
            </Link>
          </li>
        ))}
      </ul>
    </section>
  )
}
