from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple
from matplotlib import pyplot as plt

def poisson_model(x_values: Sequence[int], amplitude: float, lam: float, offset: float):
	"""Shifted Poisson model A * e^-lambda * lambda^(x-offset) / (x-offset)!."""

	try:
		import numpy as np
		from scipy.special import gammaln
	except ImportError as exc:
		raise ImportError("This script requires numpy and scipy to run the fit.") from exc

	x_array = np.asarray(x_values, dtype=float)
	shifted_x = x_array - offset
	if lam <= 0 or np.any(shifted_x < 0):
		return np.full_like(x_array, np.nan, dtype=float)

	return amplitude * np.exp(shifted_x * np.log(lam) - lam - gammaln(shifted_x + 1.0))


def read_xy_data(file_path: Path) -> Tuple[List[float], List[float]]:
	"""Read two numeric columns from a whitespace- or comma-separated file."""

	x_values: List[int] = []
	y_values: List[float] = []

	with file_path.open("r", encoding="utf-8") as handle:
		for line_number, raw_line in enumerate(handle, start=1):
			line = raw_line.strip()
			if not line or line.startswith("#"):
				continue

			parts = line.replace(",", " ").split()
			if len(parts) < 2:
				raise ValueError(
					f"Line {line_number} in {file_path} does not contain two columns."
				)

			try:
				x_value = int(parts[0]) - 2
				y_value = float(parts[1])
			except ValueError as exc:
				raise ValueError(
					f"Line {line_number} in {file_path} contains non-numeric data: {line!r}"
				) from exc

			x_values.append(x_value)
			y_values.append(y_value)

	if not x_values:
		raise ValueError(f"No data rows were found in {file_path}.")

	return x_values, y_values


def fit_poisson(x_values: Sequence[int], y_values: Sequence[float]) -> Tuple[float, float, float, List[float]]:
	"""Fit A, lambda, and offset in the shifted Poisson model using SciPy curve fitting."""

	try:
		from scipy.optimize import curve_fit
	except ImportError as exc:
		raise ImportError("This script requires scipy to run the fit.") from exc

	if not x_values:
		raise ValueError("No x values were provided.")

	initial_amplitude = max(y_values)
	initial_lambda = max(1e-6, sum(x_values) / len(x_values))
	initial_offset = 0.0
	max_offset = float(min(x_values))

	params, _ = curve_fit(
		poisson_model,
		x_values,
		y_values,
		p0=(initial_amplitude, initial_lambda, initial_offset),
		bounds=([0.0, 0.0, 0.0], [float("inf"), float("inf"), max_offset]),
		maxfev=10000,
	)
	amplitude, lam, offset = params
	fitted_y = list(poisson_model(x_values, amplitude, lam, offset))
	return float(amplitude), float(lam), float(offset), fitted_y


def main() -> None:
    
	parser = argparse.ArgumentParser(description="Fit a Poisson model to two-column data.")
	parser.add_argument(
		"data_file",
		nargs="?",
		default="poisson_data.txt",
		help="Path to the input text file with x and y columns.",
	)
	args = parser.parse_args()

	data_path = Path(args.data_file)
	if not data_path.is_file():
		raise FileNotFoundError(f"Data file not found: {data_path}")

	x_values, y_values = read_xy_data(data_path)
	amplitude, lam, offset, fitted_y = fit_poisson(x_values, y_values)

	plt.plot(x_values, y_values, "o", label="Observed Data")
	plt.plot(x_values, fitted_y, "-", label="Fitted Poisson Model")
	plt.xlabel("x")
	plt.ylabel("y")
	plt.title("Poisson Fit")
	plt.legend()
	
	print(f"Fitted amplitude A: {amplitude:.6g}")
	print(f"Fitted lambda: {lam:.6g}")
	print(f"Fitted offset: {offset:.6g}")
	plt.show()
	
if __name__ == "__main__":
	main()


