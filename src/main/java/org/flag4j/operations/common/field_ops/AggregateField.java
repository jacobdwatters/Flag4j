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

package org.flag4j.operations.common.field_ops;


import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.util.ErrorMessages;

/**
 * This class contains several low-level methods useful for computing aggregation operations on dense/sparse tensors whose elements
 * are members of a {@link Field}.
 */
public final class AggregateField {

    private AggregateField() {
        // Hide default constructor.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Computes the sum of all entries in this tensor. This can be applied to either real dense or spase tensors.
     * @param entries Entries of the tensor.
     * @return The sum of all entries in this tensor. If {entries.length == 0}, null will be returned.
     */
    public static <T extends Field<T>> T sum(Field<T>... entries) {
        if(entries.length == 0) return null;
        Field<T> sum = entries[0];

        for(int i=0, size = entries.length; i<size; i++)
            sum = sum.add((T) entries[i]);

        return (T) sum;
    }


    /**
     * Computes the sum of all entries in this tensor. This can be applied to either real dense or spase tensors.
     * @param entries Entries of the tensor.
     * @return The sum of all entries in this tensor. If {entries.length == 0}, null will be returned.
     */
    public static <T extends Field<T>> T prod(Field<T>... entries) {
        if(entries.length == 0) return null;
        Field<T> prod = entries[0];

        for(int i=0, size = entries.length; i<size; i++)
            prod = prod.mult((T) entries[i]);

        return (T) prod;
    }
}
