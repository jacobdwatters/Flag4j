/*
 * MIT License
 *
 * Copyright (c) 2023-2024. Jacob Watters
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

package org.flag4j.linalg.operations.common.complex;

import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.util.ErrorMessages;


/**
 * <p>This class contains low-level implementations for operations which check if a complex tensor satisfies some property.</p>
 * <p>Implementations are agnostic to whether the tensor is sparse or dense.</p>
 */
public final class Complex128Properties {

    private Complex128Properties() {
        // Hide default constructor in utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Checks whether a tensor contains only real values.
     * @param entries Entries of dense tensor or non-zero entries of sparse tensor.
     * @return True if the tensor only contains real values. Returns false otherwise.
     */
    public static boolean isReal(Field<Complex128>[] entries) {
        if(entries == null) return false;

        for(Field<Complex128> entry : entries)
            if(((Complex128) entry).im != 0) return false;

        return true;
    }


    /**
     * Checks whether a tensor contains at least one non-real value.
     * @param entries Entries of dense tensor or non-zero entries of sparse tensor.
     * @return True if the tensor contains at least one non-real value. Returns false otherwise.
     */
    public static boolean isComplex(Field<Complex128>[] entries) {
        if(entries == null) return false;

        for(Field<Complex128> entry : entries)
            if(((Complex128) entry).im != 0) return true;

        return false;
    }
}
