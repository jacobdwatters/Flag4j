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

package org.flag4j.linalg.operations.common.semiring_ops;

import org.flag4j.algebraic_structures.semirings.Semiring;
import org.flag4j.util.ErrorMessages;


/**
 * This utility class provides methods useful for determining properties of semi-ring tensors.
 */
public final class SemiRingProperties {

    private SemiRingProperties() {
        // Hide default constructor in utility class.
        throw new UnsupportedOperationException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }

    /**
     * Checks if an array contains only zeros.
     * @param src Array to check if it only contains zeros.
     * @return True if the {@code src} array contains only zeros. Otherwise, returns {@code false}.
     */
    public static <T extends Semiring<T>> boolean isZeros(Semiring<T>[] src) {
        for(Semiring<T> value: src)
            if(!value.isZero()) return false;

        return true;
    }


    /**
     * Checks if an array contains only ones.
     * @param src Array to check if it only contains ones.
     * @return {@code true} if the {@code src} array contains only ones. Otherwise, returns {@code false}.
     */
    public static <T extends Semiring<T>> boolean isOnes(Semiring<T>[] src) {
        for(Semiring<T> value: src)
            if(!value.isOne()) return false;

        return true;
    }
}
