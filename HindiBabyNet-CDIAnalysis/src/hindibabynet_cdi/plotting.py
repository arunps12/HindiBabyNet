"""Reusable plotting helpers for Hindi CDI analysis outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _prepare_output_path(output_dir: str | Path, filename: str) -> Path:
	path = Path(output_dir) / filename
	path.parent.mkdir(parents=True, exist_ok=True)
	return path


def _render_no_data(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
	ax.set_title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)


def save_histogram(series: pd.Series, *, title: str, xlabel: str, output_dir: str | Path, filename: str) -> Path:
	path = _prepare_output_path(output_dir, filename)
	clean = pd.to_numeric(series, errors="coerce").dropna()
	fig, ax = plt.subplots(figsize=(8, 5))
	if not clean.empty:
		sns.histplot(clean, bins=min(20, max(5, clean.nunique())), ax=ax, color="#33658A")
	else:
		_render_no_data(ax, title, xlabel, "Count")
	if not clean.empty:
		ax.set_title(title)
		ax.set_xlabel(xlabel)
		ax.set_ylabel("Count")
	fig.tight_layout()
	fig.savefig(path, dpi=200)
	plt.close(fig)
	return path


def save_bar_chart(dataframe: pd.DataFrame, *, x: str, y: str, title: str, output_dir: str | Path, filename: str) -> Path:
	path = _prepare_output_path(output_dir, filename)
	fig, ax = plt.subplots(figsize=(9, 5))
	if dataframe.empty:
		_render_no_data(ax, title, x, y)
	else:
		sns.barplot(data=dataframe, x=x, y=y, ax=ax, color="#2A9D8F")
		ax.set_title(title)
		ax.set_xlabel(x)
		ax.set_ylabel(y)
		ax.tick_params(axis="x", rotation=20)
	fig.tight_layout()
	fig.savefig(path, dpi=200)
	plt.close(fig)
	return path


def save_horizontal_bar_chart(dataframe: pd.DataFrame, *, x: str, y: str, title: str, output_dir: str | Path, filename: str) -> Path:
	path = _prepare_output_path(output_dir, filename)
	fig, ax = plt.subplots(figsize=(10, 7))
	if dataframe.empty:
		_render_no_data(ax, title, x, y)
	else:
		sns.barplot(data=dataframe, x=x, y=y, ax=ax, color="#BC6C25", orient="h")
		ax.set_title(title)
		ax.set_xlabel(x)
		ax.set_ylabel(y)
	fig.tight_layout()
	fig.savefig(path, dpi=200)
	plt.close(fig)
	return path


def save_scatter_plot(
	dataframe: pd.DataFrame,
	*,
	x: str,
	y: str,
	title: str,
	output_dir: str | Path,
	filename: str,
	hue: str | None = None,
) -> Path:
	path = _prepare_output_path(output_dir, filename)
	fig, ax = plt.subplots(figsize=(8, 5))
	if dataframe.empty:
		_render_no_data(ax, title, x, y)
	else:
		sns.scatterplot(data=dataframe, x=x, y=y, hue=hue, ax=ax, palette="deep")
		ax.set_title(title)
		ax.set_xlabel(x)
		ax.set_ylabel(y)
	fig.tight_layout()
	fig.savefig(path, dpi=200)
	plt.close(fig)
	return path


def save_line_plot(
	dataframe: pd.DataFrame,
	*,
	x: str,
	y: str,
	title: str,
	output_dir: str | Path,
	filename: str,
	hue: str | None = None,
	xlabel: str | None = None,
	ylabel: str | None = None,
) -> Path:
	path = _prepare_output_path(output_dir, filename)
	fig, ax = plt.subplots(figsize=(9, 5))
	if dataframe.empty:
		_render_no_data(ax, title, xlabel or x, ylabel or y)
	else:
		sns.lineplot(data=dataframe, x=x, y=y, hue=hue, marker="o", ax=ax)
		ax.set_title(title)
		ax.set_xlabel(xlabel or x)
		ax.set_ylabel(ylabel or y)
		ax.tick_params(axis="x", rotation=20)
	fig.tight_layout()
	fig.savefig(path, dpi=200)
	plt.close(fig)
	return path


def save_heatmap(
	dataframe: pd.DataFrame,
	*,
	index: str,
	columns: str,
	values: str,
	title: str,
	output_dir: str | Path,
	filename: str,
	cmap: str = "YlGnBu",
) -> Path:
	path = _prepare_output_path(output_dir, filename)
	fig, ax = plt.subplots(figsize=(10, 6))
	if dataframe.empty:
		_render_no_data(ax, title, columns, index)
	else:
		pivot = dataframe.pivot_table(index=index, columns=columns, values=values, aggfunc="mean")
		pivot = pivot.apply(pd.to_numeric, errors="coerce")
		if pivot.empty or pivot.notna().sum().sum() == 0:
			_render_no_data(ax, title, columns, index)
		else:
			sns.heatmap(pivot, cmap=cmap, ax=ax)
			ax.set_title(title)
			ax.set_xlabel(columns)
			ax.set_ylabel(index)
	fig.tight_layout()
	fig.savefig(path, dpi=200)
	plt.close(fig)
	return path