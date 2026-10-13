/*
 * MIT License
 *
 * Copyright (c) 2026. Jacob Watters
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package org.flag4jv3.ndarrays;


/// Describes the backing data type of an nD-array.
public enum DType {
    BOOL(1, 1),
    INT8(1, 1),
    INT16(2, 1),
    INT32(4, 1),
    INT64(8, 1),
    FLOAT32(4, 1),
    FLOAT64(8, 1),
    COMPLEX64(4, 2),
    COMPLEX128(8, 2),
    SEMIRING(-1, 1),
    RING(-1, 1),
    FIELD(-1, 1);

    /// The number of bytes per component of the data type.
    private final int bytesPerComponent;

    /// The number of components per slot of the data type.
    private final int componentsPerSlot;

    /// The number of bytes per single element.
    private final int bytesPerSlot;

    /// Flag indicating whether the data type has a known size.
    private final boolean hasKnownSize;


    DType(int bytesPerComponent, int componentsPerSlot) {
        this.bytesPerComponent = bytesPerComponent;
        this.componentsPerSlot = componentsPerSlot;
        this.hasKnownSize = bytesPerComponent >= 0;
        this.bytesPerSlot = bytesPerComponent*componentsPerSlot;
    }


    /// Gets the number of components per slot of the data type.
    ///
    /// @return The number of components per slot of the data type.
    public int componentsPerSlot() {
        return componentsPerSlot;
    }


    /// Checks whether the data type has a known size. Since implementation of [Fields][org.flag4jv3.algebra.Field],
    /// [Rings][org.flag4jv3.algebra.Ring], [Semirings][org.flag4jv3.algebra.Semiring], etc. may have differing element
    /// sizes, the number of bytes per element is not always known.
    ///
    /// @return `true` if the data type has a known size; otherwise `false`.
    public boolean hasKnownSize() {
        return hasKnownSize;
    }


    /// Gets the number of bytes per component of the data type.
    ///
    /// <blockquote style="color: #d4aeae; background-color: #571f1f; border-left: 5px solid #f44336; padding: 10px;">
    ///     <strong>Warning:</strong> The number of bytes per component of the data type is not always known.
    ///     For example, implementations of {@link org.flag4jv3.algebra.Field Fields},
    ///     {@link org.flag4jv3.algebra.Ring Rings}, etc. may have differing component sizes. For such data types, the
    ///     size cannot be known in general and this method will throw an [UnsupportedOperationException][java.lang.UnsupportedOperationException].
    ///
    ///     Use {@link #hasKnownSize} to determine whether the number of bytes per component is known.
    /// </blockquote>
    ///
    /// @return The number of bytes per component of the data type.
    ///
    /// @throws UnsupportedOperationException if the data type has an unknown size.
    /// @see #bytesPerSlot
    public int bytesPerComponent() {
        if (!hasKnownSize) {
            throw new UnsupportedOperationException("bytes per element is unknown for " + this.name() + ".");
        }

        return bytesPerComponent;
    }


    /// Gets the number of bytes per slot of the data type. This is equal to [#componentsPerSlot()]`*`[#bytesPerComponent()]
    ///
    /// <blockquote style="color: #d4aeae; background-color: #571f1f; border-left: 5px solid #f44336; padding: 10px;">
    ///     <strong>Warning:</strong> The number of bytes per slot of the data type is not always known.
    ///     For example, implementations of {@link org.flag4jv3.algebra.Field Fields},
    ///     {@link org.flag4jv3.algebra.Ring Rings}, etc. may have differing slot sizes. For such data types, the
    ///     size cannot be known in general and this method will throw an [UnsupportedOperationException][java.lang.UnsupportedOperationException].
    ///
    ///     Use {@link #hasKnownSize} to determine whether the number of bytes per component is known.
    /// </blockquote>
    ///
    /// @return The number of bytes per slot of the data type.
    ///
    /// @throws UnsupportedOperationException if the data type has an unknown size.
    /// @see #bytesPerComponent
    public int bytesPerSlot() {
        if (!hasKnownSize) {
            throw new UnsupportedOperationException("bytes per element is unknown for " + this.name() + ".");
        }

        return bytesPerSlot;
    }
}
