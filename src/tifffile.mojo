"""Compute kernels used by the Python TIFF reader and writer."""

from std.sys.info import num_physical_cores, simd_width_of as simdwidthof

comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime U16Ptr = UnsafePointer[UInt16, AnyOrigin[mut=True]]
comptime U32Ptr = UnsafePointer[UInt32, AnyOrigin[mut=True]]
comptime U64Ptr = UnsafePointer[UInt64, AnyOrigin[mut=True]]
comptime PARALLEL_PREDICTOR_BYTES = 4 * 1024 * 1024
comptime MAX_PREDICTOR_WORKERS = 8


def read_word(p: BPtr, offset: Int, size: Int, little: Bool) -> UInt64:
    var value = UInt64(0)
    if little:
        for j in range(size):
            value |= UInt64(p[offset + j]) << UInt64(8 * j)
    else:
        for j in range(size):
            value = (value << 8) | UInt64(p[offset + j])
    return value


def write_word(p: BPtr, offset: Int, size: Int, little: Bool, value: UInt64):
    if little:
        for j in range(size):
            p[offset + j] = UInt8((value >> UInt64(8 * j)) & 255)
    else:
        for j in range(size):
            p[offset + size - 1 - j] = UInt8((value >> UInt64(8 * j)) & 255)


def predictor_rows(
    address: Int,
    y0: Int,
    y1: Int,
    row_values: Int,
    samples: Int,
    itemsize: Int,
    decode: Bool,
    little: Bool,
):
    var p = BPtr(unsafe_from_address=address)
    var row_bytes = row_values * itemsize
    if little and itemsize == 1:
        if decode:
            for y in range(y0, y1):
                var base = y * row_values
                for x in range(samples, row_values):
                    p[base + x] += p[base + x - samples]
        else:
            comptime W = simdwidthof[DType.float64]()
            for y in range(y0, y1):
                var base = y * row_values
                var x = row_values
                while x - W >= samples:
                    x -= W
                    var current = p.load[width=W, alignment=1](base + x)
                    var previous = p.load[width=W, alignment=1](
                        base + x - samples
                    )
                    p.store(base + x, current - previous)
                while x > samples:
                    x -= 1
                    p[base + x] -= p[base + x - samples]
        return
    if little and itemsize == 2:
        var q = U16Ptr(unsafe_from_address=address)
        if decode:
            for y in range(y0, y1):
                var base = y * row_values
                for x in range(samples, row_values):
                    q[base + x] += q[base + x - samples]
        else:
            comptime W = simdwidthof[DType.float64]()
            for y in range(y0, y1):
                var base = y * row_values
                var x = row_values
                while x - W >= samples:
                    x -= W
                    var current = q.load[width=W, alignment=1](base + x)
                    var previous = q.load[width=W, alignment=1](
                        base + x - samples
                    )
                    q.store(base + x, current - previous)
                while x > samples:
                    x -= 1
                    q[base + x] -= q[base + x - samples]
        return
    if little and itemsize == 4:
        var q = U32Ptr(unsafe_from_address=address)
        if decode:
            for y in range(y0, y1):
                var base = y * row_values
                for x in range(samples, row_values):
                    q[base + x] += q[base + x - samples]
        else:
            comptime W = simdwidthof[DType.float64]()
            for y in range(y0, y1):
                var base = y * row_values
                var x = row_values
                while x - W >= samples:
                    x -= W
                    var current = q.load[width=W, alignment=1](base + x)
                    var previous = q.load[width=W, alignment=1](
                        base + x - samples
                    )
                    q.store(base + x, current - previous)
                while x > samples:
                    x -= 1
                    q[base + x] -= q[base + x - samples]
        return
    if little and itemsize == 8:
        var q = U64Ptr(unsafe_from_address=address)
        if decode:
            for y in range(y0, y1):
                var base = y * row_values
                for x in range(samples, row_values):
                    q[base + x] += q[base + x - samples]
        else:
            comptime W = simdwidthof[DType.float64]()
            for y in range(y0, y1):
                var base = y * row_values
                var x = row_values
                while x - W >= samples:
                    x -= W
                    var current = q.load[width=W, alignment=1](base + x)
                    var previous = q.load[width=W, alignment=1](
                        base + x - samples
                    )
                    q.store(base + x, current - previous)
                while x > samples:
                    x -= 1
                    q[base + x] -= q[base + x - samples]
        return
    if decode:
        for y in range(y0, y1):
            var base = y * row_bytes
            for x in range(samples, row_values):
                var at = base + x * itemsize
                var prev = base + (x - samples) * itemsize
                var value = read_word(p, at, itemsize, little)
                value += read_word(p, prev, itemsize, little)
                write_word(p, at, itemsize, little, value)
    else:
        for y in range(y0, y1):
            var base = y * row_bytes
            var x = row_values - 1
            while x >= samples:
                var at = base + x * itemsize
                var prev = base + (x - samples) * itemsize
                var value = read_word(p, at, itemsize, little)
                value -= read_word(p, prev, itemsize, little)
                write_word(p, at, itemsize, little, value)
                x -= 1


def predictor_copy_rows(
    src_address: Int,
    dst_address: Int,
    y0: Int,
    y1: Int,
    row_values: Int,
    samples: Int,
    itemsize: Int,
    decode: Bool,
    little: Bool,
):
    var src = BPtr(unsafe_from_address=src_address)
    var dst = BPtr(unsafe_from_address=dst_address)
    var row_bytes = row_values * itemsize
    if little and itemsize == 1:
        if decode:
            for y in range(y0, y1):
                var base = y * row_values
                for x in range(samples):
                    dst[base + x] = src[base + x]
                for x in range(samples, row_values):
                    dst[base + x] = src[base + x] + dst[base + x - samples]
        else:
            comptime W = simdwidthof[DType.float64]()
            for y in range(y0, y1):
                var base = y * row_values
                for x in range(samples):
                    dst[base + x] = src[base + x]
                var x = samples
                while x + W <= row_values:
                    dst.store(
                        base + x,
                        src.load[width=W, alignment=1](base + x)
                        - src.load[width=W, alignment=1](
                            base + x - samples
                        ),
                    )
                    x += W
                while x < row_values:
                    dst[base + x] = src[base + x] - src[base + x - samples]
                    x += 1
        return
    if little and itemsize == 2:
        var s = U16Ptr(unsafe_from_address=src_address)
        var d = U16Ptr(unsafe_from_address=dst_address)
        if decode:
            for y in range(y0, y1):
                var base = y * row_values
                for x in range(samples):
                    d[base + x] = s[base + x]
                for x in range(samples, row_values):
                    d[base + x] = s[base + x] + d[base + x - samples]
        else:
            comptime W = simdwidthof[DType.float64]()
            for y in range(y0, y1):
                var base = y * row_values
                for x in range(samples):
                    d[base + x] = s[base + x]
                var x = samples
                while x + W <= row_values:
                    d.store(
                        base + x,
                        s.load[width=W, alignment=1](base + x)
                        - s.load[width=W, alignment=1](
                            base + x - samples
                        ),
                    )
                    x += W
                while x < row_values:
                    d[base + x] = s[base + x] - s[base + x - samples]
                    x += 1
        return
    if little and itemsize == 4:
        var s = U32Ptr(unsafe_from_address=src_address)
        var d = U32Ptr(unsafe_from_address=dst_address)
        if decode:
            for y in range(y0, y1):
                var base = y * row_values
                for x in range(samples):
                    d[base + x] = s[base + x]
                for x in range(samples, row_values):
                    d[base + x] = s[base + x] + d[base + x - samples]
        else:
            comptime W = simdwidthof[DType.float64]()
            for y in range(y0, y1):
                var base = y * row_values
                for x in range(samples):
                    d[base + x] = s[base + x]
                var x = samples
                while x + W <= row_values:
                    d.store(
                        base + x,
                        s.load[width=W, alignment=1](base + x)
                        - s.load[width=W, alignment=1](
                            base + x - samples
                        ),
                    )
                    x += W
                while x < row_values:
                    d[base + x] = s[base + x] - s[base + x - samples]
                    x += 1
        return
    if little and itemsize == 8:
        var s = U64Ptr(unsafe_from_address=src_address)
        var d = U64Ptr(unsafe_from_address=dst_address)
        if decode:
            for y in range(y0, y1):
                var base = y * row_values
                for x in range(samples):
                    d[base + x] = s[base + x]
                for x in range(samples, row_values):
                    d[base + x] = s[base + x] + d[base + x - samples]
        else:
            comptime W = simdwidthof[DType.float64]()
            for y in range(y0, y1):
                var base = y * row_values
                for x in range(samples):
                    d[base + x] = s[base + x]
                var x = samples
                while x + W <= row_values:
                    d.store(
                        base + x,
                        s.load[width=W, alignment=1](base + x)
                        - s.load[width=W, alignment=1](
                            base + x - samples
                        ),
                    )
                    x += W
                while x < row_values:
                    d[base + x] = s[base + x] - s[base + x - samples]
                    x += 1
        return
    for y in range(y0, y1):
        var base = y * row_bytes
        for i in range(samples * itemsize):
            dst[base + i] = src[base + i]
        if decode:
            for x in range(samples, row_values):
                var at = base + x * itemsize
                var prev = base + (x - samples) * itemsize
                var value = read_word(src, at, itemsize, little)
                value += read_word(dst, prev, itemsize, little)
                write_word(dst, at, itemsize, little, value)
        else:
            for x in range(samples, row_values):
                var at = base + x * itemsize
                var prev = base + (x - samples) * itemsize
                var value = read_word(src, at, itemsize, little)
                value -= read_word(src, prev, itemsize, little)
                write_word(dst, at, itemsize, little, value)


@export("mti_predictor")
def predictor(
    address: Int,
    rows: Int,
    width: Int,
    samples: Int,
    itemsize: Int,
    decode: Int,
    little: Int,
) abi("C") -> Int:
    if rows < 0 or width < 0 or samples <= 0:
        return -1
    if itemsize != 1 and itemsize != 2 and itemsize != 4 and itemsize != 8:
        return -2
    if rows == 0 or width == 0:
        return 0
    if address == 0:
        return -3
    var row_values = width * samples
    var is_little = little != 0
    var workers = (
        min(rows, min(MAX_PREDICTOR_WORKERS, num_physical_cores()))
        if rows * row_values * itemsize >= PARALLEL_PREDICTOR_BYTES else 1
    )

    @__parameter
    def process(worker: Int):
        var y0 = worker * rows // workers
        var y1 = (worker + 1) * rows // workers
        predictor_rows(
            address,
            y0,
            y1,
            row_values,
            samples,
            itemsize,
            decode != 0,
            is_little,
        )

    for worker in range(workers):
        process(worker)
    return 0


@export("mti_predictor_copy")
def predictor_copy(
    src_address: Int,
    dst_address: Int,
    rows: Int,
    width: Int,
    samples: Int,
    itemsize: Int,
    decode: Int,
    little: Int,
) abi("C") -> Int:
    if rows < 0 or width < 0 or samples <= 0:
        return -1
    if itemsize != 1 and itemsize != 2 and itemsize != 4 and itemsize != 8:
        return -2
    if rows == 0 or width == 0:
        return 0
    if src_address == 0 or dst_address == 0:
        return -3
    var row_values = width * samples
    var is_little = little != 0
    var workers = (
        min(rows, min(MAX_PREDICTOR_WORKERS, num_physical_cores()))
        if rows * row_values * itemsize >= PARALLEL_PREDICTOR_BYTES else 1
    )

    @__parameter
    def process(worker: Int):
        var y0 = worker * rows // workers
        var y1 = (worker + 1) * rows // workers
        predictor_copy_rows(
            src_address,
            dst_address,
            y0,
            y1,
            row_values,
            samples,
            itemsize,
            decode != 0,
            is_little,
        )

    for worker in range(workers):
        process(worker)
    return 0


@export("mti_packbits_encode")
def packbits_encode(src_address: Int, size: Int, dst_address: Int, capacity: Int) abi("C") -> Int:
    if size < 0 or capacity < 0:
        return -1
    if size == 0:
        return 0
    if src_address == 0 or dst_address == 0:
        return -3
    var src = BPtr(unsafe_from_address=src_address)
    var dst = BPtr(unsafe_from_address=dst_address)
    var i = 0
    var written = 0
    while i < size:
        var run = 1
        while i + run < size and run < 128 and src[i + run] == src[i]:
            run += 1
        if run >= 3:
            if written + 2 > capacity:
                return -2
            dst[written] = UInt8(257 - run)
            dst[written + 1] = src[i]
            written += 2
            i += run
        else:
            var start = i
            i += run
            while i < size and i - start < 128:
                run = 1
                while i + run < size and run < 128 and src[i + run] == src[i]:
                    run += 1
                if run >= 3:
                    break
                if i - start + run > 128:
                    i = start + 128
                    break
                i += run
            var count = i - start
            if written + 1 + count > capacity:
                return -2
            dst[written] = UInt8(count - 1)
            written += 1
            comptime W = simdwidthof[DType.float64]()
            var copied = 0
            while copied + W <= count:
                dst.store(
                    written + copied,
                    src.load[width=W, alignment=1](start + copied),
                )
                copied += W
            while copied < count:
                dst[written + copied] = src[start + copied]
                copied += 1
            written += count
    return written


@export("mti_packbits_decode")
def packbits_decode(src_address: Int, size: Int, dst_address: Int, capacity: Int) abi("C") -> Int:
    if size < 0 or capacity < 0:
        return -1
    if size == 0:
        return 0
    if src_address == 0 or dst_address == 0:
        return -4
    var src = BPtr(unsafe_from_address=src_address)
    var dst = BPtr(unsafe_from_address=dst_address)
    var i = 0
    var written = 0
    while i < size:
        var header = Int(src[i])
        i += 1
        if header <= 127:
            var count = header + 1
            if i + count > size:
                return -3
            if written + count > capacity:
                return -2
            for j in range(count):
                dst[written + j] = src[i + j]
            i += count
            written += count
        elif header >= 129:
            var count = 257 - header
            if i >= size:
                return -3
            if written + count > capacity:
                return -2
            var value = src[i]
            i += 1
            for j in range(count):
                dst[written + j] = value
            written += count
    return written


@export("mti_reverse_bits")
def reverse_bits(address: Int, size: Int) abi("C") -> Int:
    if size < 0:
        return -1
    if size == 0:
        return 0
    if address == 0:
        return -2
    var p = BPtr(unsafe_from_address=address)
    for i in range(size):
        var x = p[i]
        x = ((x & 0x55) << 1) | ((x >> 1) & 0x55)
        x = ((x & 0x33) << 2) | ((x >> 2) & 0x33)
        p[i] = (x << 4) | (x >> 4)
    return 0
