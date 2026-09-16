import type { DataIndex, QuestionSet, Question, Answer, IndexCourse } from '../types/data'

const BASE = import.meta.env.BASE_URL

let indexPromise: Promise<DataIndex> | null = null

async function fetchJson<T>(relativePath: string): Promise<T> {
  // Данные обновляются чаще, чем сборка сайта, и имена файлов не меняются:
  // без ревалидации браузер показывает старый список билетов как текущий.
  const response = await fetch(`${BASE}data/${relativePath}`, { cache: 'no-cache' })
  if (!response.ok) {
    throw new Error(`${relativePath}: ${response.status}`)
  }
  return (await response.json()) as T
}

export function loadIndex(): Promise<DataIndex> {
  indexPromise ??= fetchJson<DataIndex>('index.json')
  return indexPromise
}

export function loadAnswer(slug: string, questionId: string): Promise<Answer> {
  return fetchJson<Answer>(`courses/${slug}/answers/${questionId}.json`)
}

export function findCourse(index: DataIndex, slug: string): IndexCourse | undefined {
  return index.courses.find((course) => course.slug === slug)
}

/** The question sets a course refers to, in the order the course lists them. */
export function courseQuestionSets(index: DataIndex, course: IndexCourse): QuestionSet[] {
  return course.question_sets
    .map((id) => index.question_sets.find((set) => set.id === id))
    .filter((set): set is QuestionSet => set !== undefined)
}

export function findQuestion(
  index: DataIndex,
  questionId: string,
): { set: QuestionSet; question: Question } | undefined {
  for (const set of index.question_sets) {
    const question = set.questions.find((item) => item.id === questionId)
    if (question) {
      return { set, question }
    }
  }
  return undefined
}

/** Course questions paired with this general question by the draft crosswalk. */
export function relatedQuestions(index: DataIndex, questionId: string): string[] {
  const related = new Set<string>()
  for (const crosswalk of index.crosswalks) {
    for (const pair of crosswalk.pairs) {
      if (pair.course === questionId) {
        pair.general.forEach((id) => related.add(id))
      } else if (pair.general.includes(questionId)) {
        related.add(pair.course)
      }
    }
  }
  return [...related].sort()
}

export function timecodeUrl(videoId: string, start: number): string {
  return `https://youtu.be/${videoId}?t=${Math.floor(start)}`
}

export function formatTimecode(seconds: number): string {
  const total = Math.floor(seconds)
  const hours = Math.floor(total / 3600)
  const minutes = Math.floor((total % 3600) / 60)
  const secs = total % 60
  const mm = String(minutes).padStart(2, '0')
  const ss = String(secs).padStart(2, '0')
  return hours > 0 ? `${hours}:${mm}:${ss}` : `${mm}:${ss}`
}
