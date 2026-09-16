"""Command line entry point: ``examprep <command>``."""

from __future__ import annotations

import typer
from rich.console import Console

from examprep.validate import validate_data

app = typer.Typer(
    add_completion=False,
    help="Генерация ответов на экзаменационные билеты по записям лекций.",
    no_args_is_help=True,
)
console = Console()

CourseOption = typer.Option(..., "--course", "-c", help="Слаг курса, напр. hps-skvorchevsky")
ForceOption = typer.Option(False, "--force", help="Пересчитать, игнорируя готовые артефакты")


def _todo(command: str, where: str) -> None:
    console.print(f"[yellow]{command}[/yellow] ещё не реализовано (этап: {where}).")
    raise typer.Exit(code=1)


@app.command()
def validate() -> None:
    """Проверить схемой наборы вопросов, crosswalk и course.json."""

    report = validate_data()
    for item in report.checked:
        console.print(f"[green]✓[/green] {item}")
    for warning in report.warnings:
        console.print(f"[yellow]![/yellow] {warning}")
    for error in report.errors:
        console.print(f"[red]✗[/red] {error}")

    if not report.checked:
        console.print("[yellow]Нечего проверять: в data/ нет входных файлов.[/yellow]")
    if not report.ok:
        raise typer.Exit(code=1)


@app.command()
def ingest(course: str = CourseOption, force: bool = ForceOption) -> None:
    """Собрать метаданные видео из sources.txt в course.json."""

    from examprep.ingest import ingest as run_ingest

    sources = run_ingest(course, force=force)
    console.print(f"[green]✓[/green] источников в course.json: {len(sources)}")
    for source in sources:
        minutes = f"{source.duration_s / 60:.0f} мин" if source.duration_s else "—"
        console.print(f"  {source.order:>2}. {source.video_id}  {minutes}  {source.title}")


@app.command()
def download(
    course: str = CourseOption,
    force: bool = ForceOption,
    limit: int | None = typer.Option(None, "--limit", help="Скачать только первые N видео"),
) -> None:
    """Скачать аудиодорожки в data/courses/<slug>/audio/."""

    from examprep.download import download as run_download

    paths = run_download(course, force=force, limit=limit)
    console.print(f"[green]✓[/green] аудиофайлов готово: {len(paths)}")


@app.command(name="prepare-audio")
def prepare_audio_command(
    course: str = CourseOption,
    force: bool = ForceOption,
    jobs: int = typer.Option(8, "--jobs", help="Сколько ffmpeg запускать параллельно"),
) -> None:
    """Сконвертировать аудио в 16 кГц WAV — иначе Whisper декодирует его сам и медленно."""

    from examprep.download import prepare_audio

    paths = prepare_audio(course, force=force, jobs=jobs)
    console.print(f"[green]✓[/green] WAV готово: {len(paths)}")


@app.command()
def transcribe(
    course: str = CourseOption,
    gpus: str | None = typer.Option(None, "--gpus", help="Номера GPU через запятую; «» — на CPU"),
    model: str | None = typer.Option(None, "--model", help="Модель Whisper, напр. large-v3"),
    force: bool = ForceOption,
    limit: int | None = typer.Option(None, "--limit", help="Обработать только первые N видео"),
    video: str | None = typer.Option(None, "--video", help="Только это видео, по его id"),
    timestamps: str = typer.Option("sequential", "--timestamps", help="sequential | word | chunk"),
    per_gpu: int = typer.Option(1, "--per-gpu", help="Сколько лекций считать на одной карте"),
    glossary_prompt: bool = typer.Option(
        False,
        "--glossary-prompt/--no-glossary-prompt",
        help="Подсказывать Whisper глоссарий (рискованно: модель распознаёт сам промпт)",
    ),
) -> None:
    """Транскрибировать подготовленный WAV через Whisper на torch."""

    from examprep.config import get_settings
    from examprep.transcribe.multi_gpu import transcribe_course
    from examprep.transcribe.whisper import cuda_device_count, resolve_device

    if gpus is None:
        device_list = get_settings().gpu_list if cuda_device_count() > 0 else []
    else:
        device_list = [int(g) for g in gpus.split(",") if g.strip()]

    device, compute_type = resolve_device("cuda" if device_list else None)
    console.print(f"устройство: [bold]{device}[/bold] ({compute_type}), GPU: {device_list or '—'}")

    done = transcribe_course(
        course,
        gpus=device_list,
        model_size=model,
        force=force,
        limit=limit,
        video_id=video,
        timestamps=timestamps,
        per_gpu=per_gpu,
        glossary_prompt=glossary_prompt,
    )
    console.print(f"[green]✓[/green] транскриптов готово за этот запуск: {len(done)}")


@app.command()
def index(
    course: str = CourseOption,
    force: bool = ForceOption,
    embeddings: bool = typer.Option(
        True, "--embeddings/--no-embeddings", help="Считать эмбеддинги"
    ),
) -> None:
    """Нарезать транскрипты на чанки и посчитать эмбеддинги."""

    from examprep.index import build_index

    chunks = build_index(course, force=force, with_embeddings=embeddings)
    console.print(f"[green]✓[/green] чанков: {chunks}")


@app.command()
def retrieve(
    course: str = CourseOption,
    question: str = typer.Option(..., "--question", "-q", help="Id вопроса, напр. s13"),
    top_k: int | None = typer.Option(None, "--top-k"),
) -> None:
    """Показать кандидатов гибридного поиска по билету (отладка этапа 4)."""

    from examprep.retrieve import search
    from examprep.store import load_question_set, question_set_paths

    text = None
    for path in question_set_paths():
        for item in load_question_set(path).questions:
            if item.id == question:
                text = item.text
    if text is None:
        console.print(f"[red]✗[/red] вопрос «{question}» не найден")
        raise typer.Exit(code=1)

    console.print(f"[bold]{question}[/bold]: {text}\n")
    for position, candidate in enumerate(search(course, text, top_k=top_k), start=1):
        chunk = candidate.chunk
        console.print(
            f"{position:>2}. [cyan]{chunk.chunk_id}[/cyan] "
            f"{chunk.start / 60:.0f}–{chunk.end / 60:.0f} мин "
            f"(dense {candidate.dense_rank or '—'}, bm25 {candidate.bm25_rank or '—'})"
        )
        console.print(f"    {chunk.text[:200]}…")


@app.command()
def answer(
    course: str = CourseOption,
    question: str | None = typer.Option(None, "--question", "-q", help="Только этот билет"),
    question_set: str | None = typer.Option(None, "--set", help="Только этот набор вопросов"),
    force: bool = ForceOption,
    top_k: int | None = typer.Option(None, "--top-k", help="Сколько кандидатов брать поиском"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Показать план, ничего не считая"),
) -> None:
    """Сгенерировать ответы на билеты."""

    from examprep.answer import run as run_answers

    answers = run_answers(
        course,
        set_id=question_set,
        question_id=question,
        force=force,
        top_k=top_k,
        dry_run=dry_run,
    )
    if dry_run:
        return

    console.print(f"[green]✓[/green] ответов сгенерировано: {len(answers)}")
    for item in answers:
        console.print(
            f"  {item.question_id}: coverage [bold]{item.coverage}[/bold], "
            f"цитат {len(item.citations)}"
        )


@app.command()
def status(
    course: str = CourseOption,
    check_llm: bool = typer.Option(
        False, "--check-llm", help="Проверить, что LLM-эндпоинт отвечает"
    ),
) -> None:
    """Сводка готовности по этапам пайплайна."""

    from rich.table import Table

    from examprep.status import course_status

    report = course_status(course, with_llm=check_llm)

    table = Table(title=f"Курс {report.slug}")
    table.add_column("Этап")
    table.add_column("Готово", justify="right")
    table.add_column("Комментарий")
    for stage in report.stages:
        mark = "[green]✓[/green]" if stage.complete else "[yellow]…[/yellow]"
        table.add_row(f"{mark} {stage.name}", f"{stage.done}/{stage.total}", stage.note)
    console.print(table)

    for set_id, counter in report.coverage.items():
        breakdown = ", ".join(f"{name}: {count}" for name, count in sorted(counter.items()))
        console.print(f"coverage [bold]{set_id}[/bold] — {breakdown}")

    if report.llm:
        console.print(f"LLM: {report.llm}")

    for error in report.errors:
        console.print(f"[red]✗[/red] {error}")


@app.command(name="export-index")
def export_index() -> None:
    """Собрать data/index.json для сайта."""

    from examprep.export_index import export_index as run_export

    index = run_export()
    answers = sum(len(c.answers) for c in index.courses)
    console.print(
        f"[green]✓[/green] index.json: курсов {len(index.courses)}, "
        f"наборов {len(index.question_sets)}, ответов {answers}"
    )


if __name__ == "__main__":
    app()
