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

package org.flag4j.core_temp.structures.rings;

import org.flag4j.core_temp.structures.fields.RealFloat32;
import org.flag4j.core_temp.structures.fields.RealFloat64;

/**
 * <p>A real number backed by a 32-bit integer number. Immutable</p>
 *
 * <p>This class wraps the primative int type.</p>
 */
public class RealInt32 implements Ring<RealInt32> {

    /**
     * Numerical value of field element.
     */
    private final int value;


    /**
     * Constructs a real 32-bit floating point number.
     * @param value Value of the integer number.
     */
    public RealInt32(int value) {
        this.value = value;
    }


    /**
     * Constructs a real 32-bit floating point number.
     * @param value Value of the 32-bit integer number.
     */
    public RealInt32(double value) {
        this.value = (int) value;
    }


    /**
     * Constructs a real 32-bit integer number by casting a real 64-bit floating point.
     * @param value Value of the 32-bit integer number.
     */
    public RealInt32(RealFloat64 value) {
        this.value = (int) value.getValue();
    }


    /**
     * Constructs a real 32-bit integer number by casting a real 64-bit floating point.
     * @param value Value of the 32-bit integer number.
     */
    public RealInt32(RealFloat32 value) {
        this.value = (int) value.getValue();
    }


    /**
     * Gets the value of this field element.
     * @return The value of this field element.
     */
    public float getValue() {
        return value;
    }


    /**
     * Sums two elements of this field (associative and commutative).
     *
     * @param b Second field element in sum.
     *
     * @return The sum of this element and {@code b}.
     */
    @Override
    public RealInt32 add(RealInt32 b) {
        return new RealInt32(this.value + b.value);
    }


    /**
     * Computes difference of two elements of this field.
     *
     * @param b Second field element in difference.
     *
     * @return The difference of this field element and {@code b}.
     */
    @Override
    public RealInt32 sub(RealInt32 b) {
        return new RealInt32(this.value - b.value);
    }


    /**
     * Multiplies two elements of this field (associative and commutative).
     *
     * @param b Second field element in product.
     *
     * @return The product of this field element and {@code b}.
     */
    @Override
    public RealInt32 mult(RealInt32 b) {
        return new RealInt32(this.value * b.value);
    }


    /**
     * <p>Computes the addative inverse for an element of this field.</p>
     *
     * <p>An element -x is an addative inverse for a filed element x if -x + x = 0 where 0 is the addative identity..</p>
     *
     * @return The additive inverse for this field element.
     */
    @Override
    public RealInt32 addInv() {
        return new RealInt32(-this.value);
    }


    /**
     * Computes the magnitude of this field element.
     *
     * @return The magniitude of this field element.
     */
    @Override
    public double mag() {
        return Math.abs(this.value);
    }


    /**
     * Evaluates the signum or sign function on a field element.
     *
     * @param a Value to evalute signum funciton on.
     * @return The output of the signum function evaluated on {@code a}.
     */
    public static RealInt32 sgn(RealInt32 a) {
        return new RealInt32(Math.signum(a.value));
    }


    /**
     * Compares this element of the ordered field with {@code b}.
     *
     * @param b Second elemetn of the ordered field.
     *
     * @return An int value:
     * <ul>
     *     <li>0 if this field element is equal to {@code b}.</li>
     *     <li>< 0 if this field element is less than {@code b}.</li>
     *     <li>> 0 if this field element is greater than {@code b}.</li>
     *     Hence, this method returns zero if and only if the two field elemetns are equal, a negative value if and only the field
     *     element it was called on is less than {@code b} and positive if and only if the field element it was called on is greater
     *     than {@code b}.
     * </ul>
     */
    @Override
    public int compareTo(RealInt32 b) {
        return Integer.compare(this.value, b.value);
    }


    /**
     * Computes the product of all entires of specified array.
     * @param values Values to compute product of.
     * @return The product of all values in {@code values}.
     */
    public static RealInt32 prod(RealInt32... values) {
        if(values == null || values.length == 0) {
            throw new IllegalArgumentException("Values cannot be null or empty");
        }

        int prod = values[0].value;

        for(int i = 1, length=values.length; i < length; i++) {
            prod *= values[i].value;
        }

        return new RealInt32(prod);
    }


    /**
     * Computes the sum of all entires of specified array.
     * @param values Values to compute product of.
     * @return The sum of all values in {@code values}.
     */
    public static RealInt32 sum(RealInt32... values) {
        if(values == null || values.length == 0) {
            throw new IllegalArgumentException("Values cannot be null or empty");
        }

        int sum = values[0].value;

        for(int i = 1, length=values.length; i < length; i++) {
            sum += values[i].value;
        }

        return new RealInt32(sum);
    }


    /**
     * Checks if an object is equal to this Field element.
     * @param b Object to compare to this Field element.
     * @return True if the objects are the same or are both {@link RealInt32}'s and have equal values.
     */
    @Override
    public boolean equals(Object b) {
        // Check for quick returns.
        if(this == b) return true;
        if(b == null) return false;
        if(b.getClass() != this.getClass()) return false;

        return this.value == ((RealInt32) b).value;
    }


    /**
     * Converts this field element to a string representation.
     * @return A string representation of this field element.
     */
    public String toString() {
        return String.valueOf(this.value);
    }
}
