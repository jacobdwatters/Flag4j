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

package org.flag4j.linalg.operations.common.ring_ops;

import org.flag4j.algebraic_structures.rings.Ring;
import org.flag4j.util.ErrorMessages;


/**
 * This utility class provides methods useful for comparing elements of a {@link Ring}.
 */
public final class CompareRing {

    private CompareRing() {
        // Hide constructor for utility class.
        throw new IllegalAccessError(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Computes the maximum absolute value in the specified array. This is done according to the ordering imposed by
     * {@link Double#compareTo(Double) Double.compareTo(x.mag(), y.mag())} where x and y are elements of {@code values}.
     * @param values Values to commute maximum of.
     * @return The maximum value in {@code values}. If {@code values.length} equals zero, then {@link Double#NaN} is returned.
     */
    public static <T extends Ring<T>> double maxAbs(Ring<T>... values) {
        if(values.length == 0) return Double.NaN;
        double max = values[0].mag();

        for (int i=1, length=values.length; i < length; i++) {
            double mag = values[i].mag();
            if(Double.compare(max, mag) < 0) max = mag;
        }

        return max;
    }


    /**
     * Computes the minimum absolute value in the specified array. This is done according to the ordering imposed by
     * {@link Double#compareTo(Double) Double.compareTo(x.mag(), y.mag())} where x and y are elements of {@code values}.
     * @param values Values to compute minimum of.
     * @return The minimum value in {@code values}. If {@code values.length} equals zero, then {@link Double#NaN} is returned.
     */
    public static <T extends Ring<T>> double minAbs(Ring<T>... values) {
        if(values.length == 0) return Double.NaN;
        double min = values[0].mag();

        for (int i=1, length=values.length; i < length; i++) {
            double mag = values[i].mag();
            if(Double.compare(mag, min) < 0) min = mag;
        }

        return min;
    }


    /**
     * Computes the index of the maximum absolute value in the specified array.
     * @param values Values for which compute index of maximum absolute value.
     * @return The index of the maximum absolute value in {@code values}. If the maximum absolute value occurs more than once, the
     * index of the first occurrence is returned. If {@code values.length} equals zero, then {@code -1} is returned.
     */
    public static <T extends Ring<T>> int argmaxAbs(Ring<T>... values) {
        if(values.length == 0) return -1;
        double max = values[0].mag();
        int maxdex = 0;

        for (int i=1, length=values.length; i < length; i++) {
            double mag = values[i].mag();
            if(max < mag) {
                max = mag;
                maxdex = i;
            }
        }

        return maxdex;
    }


    /**
     * Computes the index of the minimum absolute value in the specified array.
     * @param values Values for which compute index of the minimum absolute value.
     * @return The index of the minimum absolute value in {@code values}. If the minimum absolute value occurs more than once,
     * the index of the first occurrence is returned. If {@code values.length} equals zero, then {@code -1} is returned.
     */
    public static <T extends Ring<T>> int argminAbs(Ring<T>... values) {
        if(values.length == 0) return -1;
        double min = values[0].mag();
        int mindex = 0;

        for (int i=1, length=values.length; i < length; i++) {
            double mag = values[i].mag();
            if(mag < min) {
                min = mag;
                mindex = i;
            }
        }

        return mindex;
    }
}
