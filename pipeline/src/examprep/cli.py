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


@app.command()
def transcribe(
    course: str = CourseOption,
    gpus: str | None = typer.Option(None, "--gpus", help="Номера GPU через запятую; «» — на CPU"),
    model: str | None = typer.Option(None, "--model", help="Модель Whisper, напр. large-v3"),
    force: bool = ForceOption,
    limit: int | None = typer.Option(None, "--limit", help="Обработать только первые N видео"),
) -> None:
    """Транскрибировать аудио через faster-whisper."""

    from examprep.config import get_settings
    from examprep.transcribe.multi_gpu import transcribe_course
    from examprep.transcribe.whisper import cuda_device_count, resolve_device

    if gpus is None:
        device_list = get_settings().gpu_list if cuda_device_count() > 0 else []
    else:
        device_list = [int(g) for g in gpus.split(",") if g.strip()]

    device, compute_type = resolve_device("cuda" if device_list else None)
    console.print(f"устройство: [bold]{device}[/bold] ({compute_type}), GPU: {device_list or '—'}")

    done = transcribe_course(course, gpus=device_list, model_size=model, force=force, limit=limit)
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
def answer(course: str = CourseOption) -> None:
    """Сгенерировать ответы на билеты."""

    _todo("answer", "5")


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
