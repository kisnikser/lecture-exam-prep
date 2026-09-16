import { describe, expect, it } from 'vitest'

import { formatTimecode, relatedQuestions, timecodeUrl } from './data'
import type { DataIndex } from '../types/data'

const index = {
  schema_version: 1,
  courses: [],
  question_sets: [],
  crosswalks: [
    {
      schema_version: 1,
      id: 'crosswalk-demo',
      status: 'draft',
      pairs: [{ course: 's13', general: ['g16', 'g17'] }],
    },
  ],
  generated_at: '2026-09-16T00:00:00Z',
} as unknown as DataIndex

describe('relatedQuestions', () => {
  it('maps a course question to its general pair', () => {
    expect(relatedQuestions(index, 's13')).toEqual(['g16', 'g17'])
  })

  it('maps a general question back to the course question', () => {
    expect(relatedQuestions(index, 'g16')).toEqual(['s13'])
  })

  it('returns nothing for an unpaired question', () => {
    expect(relatedQuestions(index, 'g02')).toEqual([])
  })
})

describe('formatTimecode', () => {
  it('drops the hour part below an hour', () => {
    expect(formatTimecode(610)).toBe('10:10')
  })

  it('keeps the hour part above an hour', () => {
    expect(formatTimecode(3671)).toBe('1:01:11')
  })
})

describe('timecodeUrl', () => {
  it('points at a whole second', () => {
    expect(timecodeUrl('abc123', 610.7)).toBe('https://youtu.be/abc123?t=610')
  })
})
