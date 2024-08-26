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


import org.flag4j.core_temp.structures.fields.Field;
import org.flag4j.util.ErrorMessages;

/**
 * A utility class for making coparisions between {@link Field} elements.
 */
public final class CompareField {

    private CompareField() {
        // Hide constructor for utility class.
        throw new IllegalAccessError(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Computes the maximum value in the specified array. This is done according to the ordering imposed by
     * {@link Field#compareTo(Field)}.
     * @param values Values to copmute maximum of.
     * @return The maximum value in {@code values}. If {@code values.length} equals zero, then {@code null} is returned.
     */
    public static <T extends Field<T>> T max(T... values) {
        if(values.length == 0) return null;
        T max = values[0];

        for (int i=1, length=values.length; i < length; i++)
            if(max.compareTo(values[i]) < 0) max = values[i];

        return max;
    }


    /**
     * Computes the minimum value in the specified array. This is done according to the ordering imposed by
     * {@link Field#compareTo(Field)}.
     * @param values Values to copmute minimum of.
     * @return The minimum value in {@code values}. If {@code values.length} equals zero, then {@code null} is returned.
     */
    public static <T extends Field<T>> T min(T... values) {
        if(values.length == 0) return null;
        T min = values[0];

        for (int i=1, length=values.length; i < length; i++)
            if(values[i].compareTo(min) < 0) min = values[i];

        return min;
    }


    /**
     * Computes the maxcimum absolute value in the specified array. This is done according to the ordering imposed by
     * {@link Double#compareTo(Double) Double.compareTo(x.mag(), y.mag())} where x and y are elements of {@code values}.
     * @param values Values to copmute maximum of.
     * @return The maximum value in {@code values}. If {@code values.length} equals zero, then {@link Double#NaN} is returned.
     */
    public static <T extends Field<T>> double maxAbs(T... values) {
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
     * @param values Values to copmute minimum of.
     * @return The minimum value in {@code values}. If {@code values.length} equals zero, then {@link Double#NaN} is returned.
     */
    public static <T extends Field<T>> double minAbs(T... values) {
        if(values.length == 0) return Double.NaN;
        double min = values[0].mag();

        for (int i=1, length=values.length; i < length; i++) {
            double mag = values[i].mag();
            if(Double.compare(mag, min) < 0) min = mag;
        }

        return min;
    }


    /**
     * Computes the index of the maximum value in the specified array. This is done according to the ordering imposed by
     * {@link Field#compareTo(Field)}.
     * @param values Values for which copmute index of maximum value.
     * @return The index of the maximum value in {@code values}. If the maximum value occurs more than once, the index of the first
     * occuracne is returned. If {@code values.length} equals zero, then {@code -1} is returned.
     */
    public static <T extends Field<T>> int argmax(T... values) {
        if(values.length == 0) return -1;
        T max = values[0];
        int maxdex = 0;

        for (int i=1, length=values.length; i < length; i++) {
            if(max.compareTo(values[i]) < 0) {
                max = values[i];
                maxdex = i;
            }
        }

        return maxdex;
    }


    /**
     * Computes the index of the minimum value in the specified array. This is done according to the ordering imposed by
     * {@link Field#compareTo(Field)}.
     * @param values Values for which copmute index of the minimum value.
     * @return The index of the minimum value in {@code values}. If the minimum value occurs more than once, the index of the first
     * occuracne is returned. If {@code values.length} equals zero, then {@code -1} is returned.
     */
    public static <T extends Field<T>> int argmin(T... values) {
        if(values.length == 0) return -1;
        T min = values[0];
        int mindex = 0;

        for (int i=1, length=values.length; i < length; i++) {
            if(values[i].compareTo(min) < 0) {
                min = values[i];
                mindex = i;
            }
        }

        return mindex;
    }


    /**
     * Computes the index of the maximum absolute value in the specified array.
     * @param values Values for which copmute index of maximum absolute value.
     * @return The index of the maximum absolute value in {@code values}. If the maximum absollute value occurs more than once, the
     * index of the first occuracne is returned. If {@code values.length} equals zero, then {@code -1} is returned.
     */
    public static <T extends Field<T>> int argmaxAbs(T... values) {
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
     * @param values Values for which copmute index of the minimum absolute value.
     * @return The index of the minimum absolute value in {@code values}. If the minimum absolute value occurs more than once,
     * the index of the first occuracne is returned. If {@code values.length} equals zero, then {@code -1} is returned.
     */
    public static <T extends Field<T>> int argminAbs(T... values) {
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


    /**
     * Checks if an array contains only zeros.
     * @param src Array to check if it only contains zeros.
     * @return True if the {@code src} array contains only zeros.
     */
    public static boolean isZeros(Field[] src) {
        for(Field value : src) {
            if(!value.isZero()) return false;
        }

        return true;
    }
}