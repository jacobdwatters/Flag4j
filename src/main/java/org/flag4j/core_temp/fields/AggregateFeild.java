/*
 * MIT License
 *
 * Copyright (c) 2024. Jacob Watters
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

package org.flag4j.core_temp.fields;

import org.flag4j.util.ErrorMessages;

import static org.flag4j.core_temp.fields.TransformField.reduce;

/**
 * A utility class for aggregating arrays of {@link Field} elements.
 */
public final class AggregateFeild {

    private AggregateFeild() {
        // Hide constructor for utility class.
        throw new IllegalAccessError(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Computes the product of all entires of specified array.
     * @param values Values to compute product of.
     * @return The product of all values in {@code values}.
     */
    public static <T extends Field<T>> T prod(T... values) {
        return reduce(values, (T a, T b) -> a.mult(b));
    }


    /**
     * Sums all entries in the specified array.
     * @param values Values to sum.
     * @return The sum of all entries in {@code values}.
     */
    public static <T extends Field<T>> T sum(T... values) {
        return reduce(values, (T a, T b) -> a.add(b));
    }
}
