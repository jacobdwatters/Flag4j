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

package org.flag4j.algebraic_structures;

import org.flag4j.util.ErrorMessages;

import java.math.BigDecimal;
import java.math.RoundingMode;

/**
 * <p>Represents an immutable real number backed by a 32-bit floating-point value.
 *
 * <p>This class wraps the primitive {@code float} type and provides arithmetic operations
 * consistent with a mathematical field. It implements the {@link Field} interface,
 * allowing it to be used in generic algorithms that operate on field elements.
 *
 * @see Real64
 */
public class Real32 implements Field<Real32> {
    private static final long serialVersionUID = 1L;

    /**
     * The numerical value {@code -1.0}.
     */
    public final static Real32 NEGATIVE_ONE = new Real32(-1f);
    /**
     * The numerical value {@code 0.0}.
     */
    public final static Real32 ZERO = new Real32(0f);
    /**
     * The numerical value {@code 1.0}.
     */
    public final static Real32 ONE = new Real32(1f);
    /**
     * The numerical value {@code 2.0}.
     */
    public final static Real32 TWO = new Real32(2f);
    /**
     * The numerical value {@code 3.0}.
     */
    public final static Real32 THREE = new Real32(3f);
    /**
     * The numerical value {@code 10.0}.
     */
    public final static Real32 TEN = new Real32(10f);
    /**
     * The numerical value representing the square root of two.
     */
    public final static Real32 ROOT_TWO = new Real32((float) Math.sqrt(2d));
    /**
     * The numerical value representing the square root of three.
     */
    public final static Real32 ROOT_THREE = new Real32((float) Math.sqrt(3d));
    /**
     * The numerical value representing pi (π), the ratio of the circumference of a circle to its diameter.
     */
    public final static Real32 PI = new Real32((float) Math.PI);
    

    /**
     * Numerical value of field element.
     */
    private final float value;


    /**
     * Constructs a real 32-bit floating point number.
     * @param value Value of the 32-bit floating point number.
     */
    public Real32(float value) {
        this.value = value;
    }


    /**
     * Constructs a real 32-bit floating point number by casting a real 64-bit floating point.
     * @param value Value of the 32-bit floating point number.
     */
    public Real32(Real64 value) {
        this.value = (float) value.getValue();
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
    public Real32 add(Real32 b) {
        return new Real32(this.value + b.value);
    }


    /**
     * Computes difference of two elements of this field.
     *
     * @param b Second field element in difference.
     *
     * @return The difference of this field element and {@code b}.
     */
    @Override
    public Real32 sub(Real32 b) {
        return new Real32(this.value - b.value);
    }


    /**
     * Multiplies two elements of this field (associative and commutative).
     *
     * @param b Second field element in product.
     *
     * @return The product of this field element and {@code b}.
     */
    @Override
    public Real32 mult(Real32 b) {
        return new Real32(this.value * b.value);
    }


    /**
     * <p>Checks if this value is an additive identity for this semiring.
     *
     * <p>An element 0 is an additive identity if a + 0 = a for any a in the semiring.
     *
     * @return True if this value is an additive identity for this semiring. Otherwise, false.
     */
    @Override
    public boolean isZero() {
        return equals(ZERO);
    }


    /**
     * <p>Checks if this value is a multiplicative identity for this semiring.
     *
     * <p>An element 1 is a multiplicative identity if a * 1 = a for any a in the semiring.
     *
     * @return True if this value is a multiplicative identity for this semiring. Otherwise, false.
     */
    @Override
    public boolean isOne() {
        return equals(ONE);
    }


    /**
     * <p>Gets the additive identity for this semiring.
     *
     * <p>An element 0 is an additive identity if a + 0 = a for any a in the semiring.
     *
     * @return The additive identity for this semiring.
     */
    @Override
    public Real32 getZero() {
        return ZERO;
    }


    /**
     * <p>Gets the multiplicative identity for this semiring.
     *
     * <p>An element 1 is a multiplicative identity if a * 1 = a for any a in the semiring.
     *
     * @return The multiplicative identity for this semiring.
     */
    @Override
    public Real32 getOne() {
        return ONE;
    }


    /**
     * Computes the quotient of two elements of this field.
     * @param b Second field element in quotient.
     * @return The quotient of this field element and {@code b}.
     */
    @Override
    public Real32 div(Real32 b) {
        return new Real32(this.value / b.value);
    }


    /**
     * Sums an element of this field with a real number (associative and commutative).
     *
     * @param b Real element in sum.
     *
     * @return The sum of this element and {@code b}.
     */
    @Override
    public Real32 add(double b) {
        return new Real32((float) (value + b));
    }


    /**
     * Computes difference of an element of this field and a real number.
     *
     * @param b Real value in difference.
     *
     * @return The difference of this ring element and {@code b}.
     */
    @Override
    public Real32 sub(double b) {
        return new Real32((float) (value - b));
    }


    /**
     * Multiplies an element of this field with a real number (associative and commutative).
     *
     * @param b Real number in product.
     *
     * @return The product of this field element and {@code b}.
     */
    @Override
    public Real32 mult(double b) {
        return new Real32((float) (value * b));
    }


    /**
     * Computes the quotient of an element of this field and a real number.
     *
     * @param b Real number in quotient.
     *
     * @return The quotient of this field element and {@code b}.
     */
    @Override
    public Real32 div(double b) {
        return new Real32((float) (value / b));
    }


    /**
     * <p>Computes the multiplicative inverse for an element of this field.
     *
     * <p>An element x<sup>-1</sup> is a multiplicative inverse for a field element x if x<sup>-1</sup>*x = 1 where 1 is the
     * multiplicative identity.
     *
     * @return The multiplicative inverse for this field element.
     */
    @Override
    public Real32 multInv() {
        return new Real32(1f/this.value);
    }


    /**
     * <p>Computes the additive inverse for an element of this field.
     *
     * <p>An element -x is an additive inverse for a field element x if -x + x = 0 where 0 is the additive identity.
     *
     * @return The additive inverse for this field element.
     */
    @Override
    public Real32 addInv() {
        return new Real32(-this.value);
    }


    /**
     * Computes the magnitude of this field element.
     *
     * @return The magnitude of this field element.
     */
    @Override
    public double mag() {
        return Math.abs(this.value);
    }


    /**
     * Computes the square root of this field element.
     *
     * @return The square root of this field element.
     */
    @Override
    public Real32 sqrt() {
        return new Real32((float) Math.sqrt(this.value));
    }


    /**
     * Evaluates the signum or sign function on a field element.
     *
     * @param a Value to evaluate signum function on.
     * @return The output of the signum function evaluated on {@code a}.
     */
    public static Real32 sgn(Real32 a) {
        return new Real32(Math.signum(a.value));
    }


    /**
     * Compares this element of the ordered field with {@code b}.
     *
     * @param b Second element of the ordered field.
     *
     * @return An int value:
     * <ul>
     *     <li>0 if this field element is equal to {@code b}.</li>
     *     <li>< 0 if this field element is less than {@code b}.</li>
     *     <li>> 0 if this field element is greater than {@code b}.</li>
     *     Hence, this method returns zero if and only if the two field elements are equal, a negative value if and only the field
     *     element it was called on is less than {@code b} and positive if and only if the field element it was called on is greater
     *     than {@code b}.
     * </ul>
     */
    @Override
    public int compareTo(Real32 b) {
        return Float.compare(this.value, b.value);
    }


    /**
     * Converts this semiring value to an equivalent double value.
     *
     * @return A double value equivalent to this semiring element.
     */
    @Override
    public double doubleValue() {
        return value;
    }


    /**
     * Checks if this field element is finite in magnitude.
     *
     * @return True if this field element is finite in magnitude. False otherwise (i.e. infinite, NaN etc.).
     */
    @Override
    public boolean isFinite() {
        return Float.isFinite(this.value);
    }


    /**
     * Checks if this field element is infinite in magnitude.
     *
     * @return True if this field element is infinite in magnitude. False otherwise (i.e. finite, NaN, etc.).
     */
    @Override
    public boolean isInfinite() {
        return Float.isInfinite(this.value);
    }


    /**
     * Checks if this field element is NaN in magnitude.
     *
     * @return True if this field element is NaN in magnitude. False otherwise (i.e. finite, NaN, etc.).
     */
    @Override
    public boolean isNaN() {
        return Float.isNaN(this.value);
    }


    /**
     * Computes the product of all data of specified array.
     * @param values Values to compute product of.
     * @return The product of all values in {@code values}.
     */
    public static Real32 prod(Real32... values) {
        if(values == null || values.length == 0)
            throw new IllegalArgumentException("Values cannot be null or empty");

        float prod = values[0].value;
        for(int i = 1, length=values.length; i < length; i++)
            prod *= values[i].value;

        return new Real32(prod);
    }


    /**
     * Computes the sum of all data of specified array.
     * @param values Values to compute product of.
     * @return The sum of all values in {@code values}.
     */
    public static Real32 sum(Real32... values) {
        if(values == null || values.length == 0)
            throw new IllegalArgumentException("Values cannot be null or empty");

        float prod = values[0].value;
        for(int i = 1, length=values.length; i < length; i++)
            prod += values[i].value;

        return new Real32(prod);
    }


    /**
     * Wraps a primitive float array as a {@link Real32} array.
     * @param arr Array to wrap.
     * @return A {@link Real32} array containing the values of {@code arr}. If {@code arr==null} then {@code null} will be
     * returned.
     */
    public static Real32[] wrapArray(float... arr) {
        if(arr == null) return null;

        Real32[] wrapped = new Real32[arr.length];
        for(int i=0, size=arr.length; i<size; i++)
            wrapped[i] = new Real32(arr[i]);

        return wrapped;
    }


    /**
     * Wraps a {@link Float} array as a {@link Real32} array.
     * @param arr Array to wrap.
     * @return A {@link Real32} array containing the values of {@code arr}. If {@code arr==null} then {@code null} will be
     * returned.
     */
    public static Real32[] wrapArray(Float... arr) {
        if(arr == null) return null;

        Real32[] wrapped = new Real32[arr.length];
        for(int i=0, size=arr.length; i<size; i++)
            wrapped[i] = new Real32(arr[i]);

        return wrapped;
    }


    /**
     * Rounds number to specified number of decimal places.
     *
     * @param n Number to round.
     * @param decimals Number of decimals to round to.
     * @return The number {@code n} rounded to the specified
     * 		number of decimals.
     * @throws IllegalArgumentException If decimals is less than zero.
     */
    public static Real32 round(Real32 n, int decimals) {
        if (decimals < 0)
            throw new IllegalArgumentException(ErrorMessages.getNegValueErr(decimals));

        float value = (Double.isFinite(n.value))
                ? BigDecimal.valueOf(n.value).setScale(decimals, RoundingMode.HALF_UP).floatValue()
                : n.value;

        return new Real32(value);
    }


    /**
     * Checks if an object is equal to this MMField element.
     * @param b Object to compare to this MMField element.
     * @return True if the objects are the same or are both {@link Real32}'s and have equal values.
     */
    @Override
    public boolean equals(Object b) {
        // Check for quick returns.
        if(this == b) return true;
        if(b == null) return false;
        if(b.getClass() != this.getClass()) return false;

        return this.value == ((Real32) b).value;
    }


    @Override
    public int hashCode() {
        return Float.hashCode(value);
    }


    /**
     * Converts this field element to a string representation.
     * @return A string representation of this field element.
     */
    public String toString() {
        return String.valueOf(this.value);
    }
}
