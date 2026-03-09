"""Command-line interface for phantom QA processing."""

import importlib
import logging
import pkgutil
from pathlib import Path
from typing import get_args, get_origin

import attrs
import click

import phantomkit.analyses

logger = logging.getLogger(__name__)

# Fields that are internal to pydra or used for input routing
_SKIP_FIELDS = frozenset({"constructor"})
_INPUT_FIELD_NAMES = frozenset({"input_image", "input_images"})


# ---------------------------------------------------------------------------
# Protocol discovery
# ---------------------------------------------------------------------------


def _discover_protocols() -> dict[str, tuple]:
    """
    Scan ``phantomkit.analyses`` for (single, batch) workflow pairs.

    Returns a dict mapping a hyphenated slug (e.g. ``"vial-signal"``) to a
    ``(single_cls, batch_cls_or_None)`` tuple.  Only non-private modules that
    contain at least one pydra workflow class (detected by ``Outputs``
    attribute) are included.
    """
    found: dict[str, tuple] = {}
    for info in pkgutil.iter_modules(phantomkit.analyses.__path__):
        if info.name.startswith("_"):
            continue
        mod = importlib.import_module(f"phantomkit.analyses.{info.name}")
        for attr_name, obj in vars(mod).items():
            if attr_name.startswith("_") or not attr_name.endswith("Analysis"):
                continue
            if isinstance(obj, type) and hasattr(obj, "Outputs"):
                slug = info.name.replace("_", "-")
                batch = getattr(mod, f"{attr_name}Batch", None)
                found[slug] = (obj, batch)
                break  # one primary workflow per module
    return found


# ---------------------------------------------------------------------------
# Signature helpers
# ---------------------------------------------------------------------------


def _is_required(field: attrs.Attribute) -> bool:
    """Return True if an attrs field has no real default (pydra NOTHING)."""
    if isinstance(field.default, attrs.Factory):
        return getattr(field.default.factory, "__name__", "") == "nothing_factory"
    return field.default is attrs.NOTHING


def _workflow_fields(cls) -> list[attrs.Attribute]:
    """Return user-visible attrs fields, excluding internal pydra fields."""
    return [
        f
        for f in attrs.fields(cls)
        if not f.name.startswith("_") and f.name not in _SKIP_FIELDS
    ]


def _annotation_to_click(ann) -> tuple[click.ParamType, bool]:
    """
    Map a Python type annotation to a ``(click_type, multiple)`` pair.

    ``multiple=True`` means the option should be repeatable (for ``list[X]``).
    """
    import types as _types
    import typing

    origin = get_origin(ann)
    args = get_args(ann)

    # Union / Optional  (X | None  or  Optional[X])
    if origin is typing.Union or isinstance(ann, _types.UnionType):
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            return _annotation_to_click(non_none[0])

    # list[X]
    if origin is list:
        inner, _ = _annotation_to_click(args[0]) if args else (click.STRING, False)
        return inner, True

    # fileformats FileSet subclasses → paths
    try:
        from fileformats.core import FileSet

        if isinstance(ann, type) and issubclass(ann, FileSet):
            return click.Path(), False
    except ImportError:
        pass

    if ann is Path or (isinstance(ann, type) and issubclass(ann, Path)):
        return click.Path(), False
    if ann is int:
        return click.INT, False
    if ann is float:
        return click.FLOAT, False
    if ann is bool:
        return click.BOOL, False
    return click.STRING, False


# ---------------------------------------------------------------------------
# Dynamic command builder
# ---------------------------------------------------------------------------


def _build_command(slug: str, single_cls, batch_cls) -> click.Command:
    """Build a click Command whose options mirror the workflow's parameters."""

    single_fields = {f.name: f for f in _workflow_fields(single_cls)}

    def _callback(**kwargs):
        input_val: str = kwargs.pop("input")
        plugin: str = kwargs.pop("plugin")
        pattern: str = kwargs.pop("pattern")
        # Convert click tuples (from multiple=True) to lists
        wf_kwargs = {
            k: list(v) if isinstance(v, tuple) else v for k, v in kwargs.items()
        }
        # Remove None values for optional params (keep explicit None only if needed)
        wf_kwargs = {
            k: v
            for k, v in wf_kwargs.items()
            if v is not None
            or (k in single_fields and not _is_required(single_fields[k]))
        }

        path = Path(input_val)

        if path.is_dir():
            # Batch mode: glob for images
            if batch_cls is None:
                raise click.UsageError(
                    f"Protocol '{slug}' does not support batch mode."
                )
            images = sorted(str(p) for p in path.rglob(pattern))
            if not images:
                raise click.UsageError(
                    f"No files matching '{pattern}' found under {path}"
                )
            logger.info("Batch mode: found %d image(s):", len(images))
            for img in images:
                logger.info("  %s", img)
            wf = batch_cls(input_images=images, **wf_kwargs)
        else:
            # Single mode
            wf = single_cls(input_image=input_val, **wf_kwargs)

        from pydra.engine import Submitter

        with Submitter(plugin=plugin) as sub:
            sub(wf)
        logger.info("Done.")

    # Build click params from the workflow's attrs fields
    click_params: list[click.Parameter] = []
    for name, field in single_fields.items():
        if name in _INPUT_FIELD_NAMES:
            continue
        ann = field.type if field.type is not None else str
        click_type, multiple = _annotation_to_click(ann)
        required = _is_required(field) and not multiple
        default = None if _is_required(field) else field.default
        click_params.append(
            click.Option(
                [f"--{name.replace('_', '-')}"],
                type=click_type,
                default=default,
                required=required,
                multiple=multiple,
                show_default=default is not None,
                help=f"{name}.",
            )
        )

    # Common options
    click_params += [
        click.Option(
            ["--plugin"],
            type=click.Choice(["cf", "serial"]),
            default="cf",
            show_default=True,
            help="Pydra execution plugin.",
        ),
        click.Option(
            ["--pattern"],
            default="*.nii.gz",
            show_default=True,
            help=(
                "Glob pattern used when INPUT is a directory (batch mode) "
                "to find NIfTI images recursively."
            ),
        ),
        click.Argument(["input"]),
    ]

    # Compose docstring from workflow
    doc = getattr(single_cls, "__doc__", None) or ""
    first_line = (
        doc.strip().splitlines()[0] if doc.strip() else f"Run the {slug} protocol."
    )
    help_text = (
        f"{first_line}\n\n"
        "INPUT may be a single NIfTI file (single-session mode) or a directory "
        "(batch mode — images are found with --pattern)."
    )

    return click.Command(
        name=slug,
        callback=_callback,
        params=click_params,
        help=help_text,
    )


# ---------------------------------------------------------------------------
# Main CLI group
# ---------------------------------------------------------------------------


@click.group()
def main() -> None:
    """Phantom QA processing CLI."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )


@main.command("list")
def list_protocols() -> None:
    """List all available protocol workflows."""
    protocols = _discover_protocols()
    if not protocols:
        click.echo("No protocols found.")
        return
    click.echo("Available protocols:")
    for slug, (single_cls, batch_cls) in protocols.items():
        batch_info = " (batch supported)" if batch_cls is not None else ""
        doc = getattr(single_cls, "__doc__", "") or ""
        summary = doc.strip().splitlines()[0] if doc.strip() else ""
        click.echo(f"  {slug}{batch_info}")
        if summary:
            click.echo(f"    {summary}")


@main.group()
def run() -> None:
    """Run a phantom QA protocol workflow.

    Use ``phantomkit run <protocol> --help`` to see the options for a specific
    protocol.  Pass a single NIfTI file as INPUT for single-session mode, or a
    directory for batch mode.
    """


def _register_commands() -> None:
    protocols = _discover_protocols()
    for slug, (single_cls, batch_cls) in protocols.items():
        run.add_command(_build_command(slug, single_cls, batch_cls))


# ---------------------------------------------------------------------------
# Plot subgroup — collects the per-module click commands from phantomkit.plotting
# ---------------------------------------------------------------------------


@main.group()
def plot() -> None:
    """Generate QA plots.

    Use ``phantom-process plot <subcommand> --help`` for options.
    """


def _register_plot_commands() -> None:
    import phantomkit.plotting as _pkg

    for info in pkgutil.iter_modules(_pkg.__path__):
        if info.name.startswith("_"):
            continue
        mod = importlib.import_module(f"phantomkit.plotting.{info.name}")
        cmd = getattr(mod, "main", None)
        if isinstance(cmd, (click.Command, click.Group)):
            plot.add_command(cmd, name=info.name.replace("_", "-"))


_register_commands()
_register_plot_commands()


if __name__ == "__main__":
    main()
