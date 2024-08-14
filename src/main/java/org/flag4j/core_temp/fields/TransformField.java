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

import java.util.function.BinaryOperator;
import java.util.function.Function;


/**
 * A utility class for computing transformations on Field arrays.
 */
public final class TransformField {

    private TransformField() {
        // Hide constructor for utility class.
        throw new IllegalAccessError(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * <p>Applies a specified function to each element of the given array and returns a new array containing the results of the
     * mapping operation. This produces a new array where each element is the result of applying the provided {@code mapper}
     * function to the corresponding element in the input array.</p>
     *
     * <p>For example, if the array contains elements [a, b, c] and the {@code mapper} function multiplies each element by 2,
     * the returned array will contain [2a, 2b, 2c].</p>
     *
     * @param values the array of field elements to map; must not be null
     * @param mapper the mapping function to apply to each element; must not be null
     * @return a new array containing the results of applying the {@code mapper} function to each element in the {@code values} array
     * @throws NullPointerException if {@code values} or {@code mapper} is null
     * @throws IllegalArgumentException if {@code values} is empty
     */
    @SuppressWarnings("unchecked")
    public static <T extends Field<T>> T[] map(T[] values, Function<T, T> mapper) {
        if (values == null || mapper == null) {
            throw new NullPointerException("Values and mapper must not be null.");
        }
        if (values.length == 0) {
            throw new IllegalArgumentException("Cannot map array with length zero.");
        }

        T[] result = (T[]) java.lang.reflect.Array.newInstance(values.getClass().getComponentType(), values.length);
        for (int i = 0; i < values.length; i++) {
            result[i] = mapper.apply(values[i]);
        }

        return result;
    }


    /**
     * <p>Reduces the given array of field elements to a single element by repeatedly applying a binary operator
     * to combine elements. The reduction is performed from left to right, starting with the first element
     * in the array and hence is a "left-reduce" or "left fold".</p>
     *
     * <p>The {@code reduce} operation applies the specified {@code accumulator} function to the first
     * element and the second element, then applies it to the result and the third element, and so on
     * until all elements have been combined into a single result. The operation assumes that the array
     * contains at least one element.</p>
     *
     * <p>For example, if the array contains elements [a, b, c, d] and the accumulator is a function
     * which adds elements, the method would compute (((a + b) + c) + d).</p>
     *
     * @param values The array of field elements to reduce. Cannot be not be empty or null.
     * @param accumulator The binary operator to combine elements.
     * @return the result of lef-reducing all elements {@code values} using the specified {@code accumulator}.
     * @throws IllegalArgumentException if {@code values} is empty
     * @throws NullPointerException if {@code values} or {@code accumulator} is null
     */
    public static <T extends Field<T>> T reduce(T[] values, BinaryOperator<T> accumulator) {
        if (values == null || accumulator == null)
            throw new NullPointerException("Values and accumulator must not be null.");
        if (values.length == 0)
            throw new IllegalArgumentException("Cannot reduce array with length zero.");

        T result = values[0];
        for (int i = 1; i < values.length; i++) {
            result = accumulator.apply(result, values[i]);
        }

        return result;
    }
}
