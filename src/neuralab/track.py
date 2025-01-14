# %%
from abc import ABC
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from functools import wraps
from threading import Lock
from typing import IO, Any, BinaryIO, Callable, Literal, Optional, cast

from rich import console, progress, style, table, text
from rich.progress import BarColumn, Progress, ProgressColumn, Task
from rich.text import Text as RichText
from sympy import fu

from neuralab import logging, settings

type TaskUnit = Literal["byte", "it"]


class TaskStatsColumn(ProgressColumn):
    """A column containing text with fixed width."""

    def __init__(
        self,
        text_format: str,
        style: str = "none",
        justify: console.JustifyMethod = "left",
        markup: bool = True,
        highlighter=None,
        overflow: Optional[console.OverflowMethod] = None,
        width: int = 0,
        table_column: Optional[table.Column] = None,
    ):
        super().__init__(table_column=table_column)
        self.text_format = text_format
        self.justify = justify
        self.style = style
        self.markup = markup
        self.highlighter = highlighter
        self.overflow = overflow
        self.width = width
        super().__init__()

    def render(self, task: "Task") -> RichText:

        _text = self.text_format.format(task=task)
        if self.markup:
            text = RichText.from_markup(_text, style=self.style, justify=self.justify)
        else:
            text = RichText(_text, style=self.style, justify=self.justify)
        if self.highlighter:
            self.highlighter.highlight(text)
        if self.width:
            text.truncate(max_width=self.width, overflow=self.overflow, pad=True)
        return text


PROGRESS: Optional[Progress] = None
PROGRESS_LOCK = Lock()
PROGRESS_ARGS = [
    TaskStatsColumn(
        "{task.description}",
        table_column=table.Column(
            ratio=2,
            no_wrap=True,
            overflow="ellipsis",
        ),
    ),
    BarColumn(
        table_column=table.Column(ratio=1),
    ),
]
PROGRESS_KWARGS = dict(
    transient=True,
    expand=True,
)


@contextmanager
def _adquire_progress_context(): # future live context..
    global PROGRESS
    with PROGRESS_LOCK:
        if PROGRESS is None:
            context_owner = True
            progress = Progress(
                *PROGRESS_ARGS,
                console=logging.console,
                **PROGRESS_KWARGS,
            )
            PROGRESS = progress
        else:
            context_owner = False
            progress = PROGRESS

    if context_owner:
        with progress:
            try:
                yield progress
            except Exception:
                progress.console.print_exception()
                raise
            finally:
                with PROGRESS_LOCK:
                    PROGRESS = None
    else:
        try:
            yield progress
        except Exception:
            progress.console.print_exception()
            raise


class TaskTracker(ABC):
    context: Any
    progress: Progress
    task_id: Any

    def __init__(self, description: str, leave=True, **kwargs):
        self.description = description
        self.leave = leave
        self.task_kwargs = kwargs

    def __enter__(self):
        self.context = _adquire_progress_context()
        self.progress = self.context.__enter__()
        self.task_id = self.progress.add_task(
            description=self.description,
            **self.task_kwargs,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is None:
                self.update(
                    refresh=True,
                    visible=not self.leave,
                    completed=self.progress.tasks[self.task_id].total,
                )
            else:
                self.update(
                    description=f"[red]{exc_type}[/red]: {exc_val}",
                    refresh=True,
                    visible=True,
                )
        finally:
            self.context.__exit__(exc_type, exc_val, exc_tb)
            del self.task_id
            del self.progress
            del self.context

    def update(self, **kwargs):
        assert hasattr(self, "progress"), "TaskTracker is not active"
        self.progress.update(self.task_id, **kwargs)

    @property
    def console(self):
        assert hasattr(self, "progress"), "TaskTracker is not active"
        return self.progress.console

    def wrap_io[
        T: IO[bytes] | BinaryIO
    ](
        self,
        meth: Literal["read", "write"],
        io_obj: T,
        total: Optional[int] = None,
    ) -> T:
        io_wrapper = ObjectWrapper(io_obj)

        match meth:
            case "read":
                return self.progress.wrap_file(
                    io_obj,  # type: ignore
                    task_id=self.task_id,
                    total=total,
                )  # type: ignore

                @io_wrapper.wraps_method("read")
                def wrapped_read(*args, **kwargs):
                    data = io_obj.read(*args, **kwargs)
                    self.update(advance=len(data))
                    return data

                @io_wrapper.wraps_method("readline")
                def wrapped_readline(*args, **kwargs):
                    data = io_obj.readline(*args, **kwargs)
                    self.update(advance=len(data))
                    return data

            case "write":

                @io_wrapper.wraps_method("write")
                def wrapped_write(content, *args, **kwargs):
                    data = io_obj.write(content, *args, **kwargs)
                    self.update(advance=len(content))
                    return data

            case _:
                raise ValueError(f"Invalid method {meth}")

        return io_wrapper  # type: ignore


class ObjectWrapper:
    __slots__ = ("_inner",)

    def __init__(self, wrapped):
        object.__setattr__(self, "_inner", wrapped)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __setattr__(self, name, value):
        return setattr(self._inner, name, value)

    def wraps_method(self, meth: str):
        inner_fn = object.__getattribute__(self._inner, meth)

        def decorator(outer_fn):
            fn = wraps(inner_fn)(outer_fn)
            object.__setattr__(self._inner, "meth", fn)

        return decorator


@contextmanager
def task(*args, **kwargs):
    with TaskTracker(*args, **kwargs) as tracker:
        yield tracker


type ParallelMapFn[T, R] = Callable[[T], R]


def parallel_map[
    T, R
](
    fn: ParallelMapFn[T, R],
    items: list[T],
    *args,
    description: str,
    num_workers=settings.DEFAULT_NUM_WORKERS,
    wait_on_exc: bool = False,
    **kwargs,
) -> Any:
    _: Any = None
    results: list[R] = [_] * len(items)

    with task(total=len(items), description=description, **kwargs) as trackbar:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                # NOTE: we need to use lambda to capture the current value of `p` and `i`
                executor.submit(lambda p, i: (p, fn(i, *args)), *pi)
                for pi in enumerate(items)
            ]

            try:
                for result in as_completed(futures):
                    if exc := result.exception():
                        raise exc

                    pos, res = result.result()                    
                    results[pos] = res
                    trackbar.update(advance=1)
                    
            except:
                executor.shutdown(wait=wait_on_exc, cancel_futures=True)
                raise

    return results


if __name__ == "__main__":
    from time import sleep

    def calc_square(n: int) -> int:

        if n == 4:
            return parallel_map(
                calc_square,
                [1, 2],
                item_type=int,
                description="re Calculating squares",
            )

        with task(
            total=n,
            description=f"Calculating {n}",
            transient=True,
        ) as trackbar:
            for n in range(n):
                sleep(1)
                trackbar.update(advance=1)

                if n == 5:
                    ...
                    # raise Exception("Error")

            return n * n

    result = parallel_map(
        calc_square,
        [1, 2, 3, 4, 8],
        item_type=int,
        description="Calculating squares",
        transient=True,
    )
    print(result)

# %%
