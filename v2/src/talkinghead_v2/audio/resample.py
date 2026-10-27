"""Bounded-history rational FIR reference, not a real-time optimized backend.

Output j samples source time j*input_rate/output_rate. A centered, windowed
sinc retains history and waits for right context. EOF uses zero extension only
inside the filter, emitting exactly ceil(N*out/in) sample timestamps below N/in.
"""

from fractions import Fraction
import math

from talkinghead_v2.contracts import ContractError, integer, numeric_tuple


class RationalResampler:
    def __init__(self, input_rate: int, output_rate: int, *, radius: int = 32):
        integer(input_rate, "input_rate", 1)
        integer(output_rate, "output_rate", 1)
        integer(radius, "radius", 2)
        self.input_rate = input_rate
        self.output_rate = output_rate
        self.radius = radius
        self.input_count = 0
        self.output_count = 0
        self._origin = 0
        self._buffer: list[float] = []
        self._kernels: dict[int, tuple[float, ...]] = {}
        self.closed = False

    @property
    def valid_end_sample(self) -> Fraction:
        return Fraction(self.input_count * self.output_rate, self.input_rate)

    @property
    def phase(self) -> Fraction:
        return Fraction((self.output_count * self.input_rate) % self.output_rate, self.output_rate)

    @property
    def buffered_samples(self) -> int:
        return len(self._buffer)

    @property
    def right_context(self) -> Fraction:
        return Fraction(self.radius, self.input_rate) if self.input_rate != self.output_rate else Fraction(0)

    def push(self, samples: tuple[float, ...]) -> tuple[float, ...]:
        if self.closed:
            raise ContractError("resampler is closed")
        numeric_tuple(samples, "samples")
        if self.input_rate == self.output_rate:
            self.input_count += len(samples)
            self.output_count += len(samples)
            return samples
        self._buffer.extend(samples)
        self.input_count += len(samples)
        return self._emit(final=False)

    def finish(self) -> tuple[float, ...]:
        if self.closed:
            return ()
        self.closed = True
        result = self._emit(final=True) if self.input_rate != self.output_rate else ()
        self._buffer.clear()
        self._origin = self.input_count
        return result

    def cancel(self) -> None:
        """Discard filter state; cancellation must never flush a playable tail."""
        self.closed = True
        self._buffer.clear()
        self._origin = self.input_count

    def _kernel(self, phase: int) -> tuple[float, ...]:
        if phase not in self._kernels:
            fraction = phase / self.output_rate
            cutoff = min(1.0, self.output_rate / self.input_rate) * 0.94
            weights = []
            for offset in range(-self.radius + 1, self.radius + 1):
                distance = offset - fraction
                z = cutoff * distance
                sinc = 1.0 if z == 0 else math.sin(math.pi*z) / (math.pi*z)
                window = 0.5 + 0.5*math.cos(math.pi*distance/self.radius) if abs(distance) < self.radius else 0.0
                weights.append(cutoff * sinc * window)
            total = sum(weights)
            self._kernels[phase] = tuple(weight / total for weight in weights)
        return self._kernels[phase]

    def _emit(self, *, final: bool) -> tuple[float, ...]:
        result = []
        limit = (self.input_count*self.output_rate + self.input_rate - 1) // self.input_rate
        while self.output_count < limit:
            center, phase = divmod(self.output_count * self.input_rate, self.output_rate)
            if not final and center + self.radius >= self.input_count:
                break
            first = center - self.radius + 1
            value = 0.0
            for offset, weight in enumerate(self._kernel(phase)):
                source = first + offset
                if 0 <= source < self.input_count:
                    if source < self._origin:
                        raise RuntimeError("resampler history invariant failed")
                    value += self._buffer[source-self._origin] * weight
            result.append(value)
            self.output_count += 1
        next_center = (self.output_count*self.input_rate) // self.output_rate
        keep_from = min(self.input_count, max(0, next_center-self.radius+1))
        remove = max(0, keep_from-self._origin)
        del self._buffer[:remove]
        self._origin += remove
        return tuple(result)
