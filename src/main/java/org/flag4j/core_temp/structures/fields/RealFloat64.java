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

package org.flag4j.core_temp.structures.fields;


/**
 * <p>A real number backed by a 64-bit floating point number. Immutable</p>
 *
 * <p>This class wraps the primitive double type.</p>
 */
public class RealFloat64 implements Field<RealFloat64> {


    /**
     * The numerical value 1.0.
     */
    public final static RealFloat64 ONE = new RealFloat64(1);
    /**
     * The numerical value 0.0.
     */
    public final static RealFloat64 ZERO = new RealFloat64(0);


    /**
     * Numerical value of field element.
     */
    private final double value;


    /**
     * Constructs a real 32-bit floating point number.
     * @param value Value of the 32-bit floating point number.
     */
    public RealFloat64(RealFloat32 value) {
        this.value = value.getValue();
    }


    /**
     * Constructs a real 32-bit floating point number.
     * @param value Value of the 32-bit floating point number.
     */
    public RealFloat64(double value) {
        this.value = value;
    }


    /**
     * Gets the value of this field element.
     * @return The value of this field element.
     */
    public double getValue() {
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
    public RealFloat64 add(RealFloat64 b) {
        return new RealFloat64(this.value + b.value);
    }


    /**
     * Computes difference of two elements of this field.
     *
     * @param b Second field element in difference.
     *
     * @return The difference of this field element and {@code b}.
     */
    @Override
    public RealFloat64 sub(RealFloat64 b) {
        return new RealFloat64(this.value - b.value);
    }


    /**
     * Multiplies two elements of this field (associative and commutative).
     *
     * @param b Second field element in product.
     *
     * @return The product of this field element and {@code b}.
     */
    @Override
    public RealFloat64 mult(RealFloat64 b) {
        return new RealFloat64(this.value * b.value);
    }


    /**
     * <p>Checks if this value is an additive identity for this semi-ring.</p>
     *
     * <p>An element 0 is an additive identity if a + 0 = a for any a in the semi-ring.</p>
     *
     * @return True if this value is an additive identity for this semi-ring. Otherwise, false.
     */
    @Override
    public boolean isZero() {
        return equals(ZERO);
    }


    /**
     * <p>Checks if this value is a multiplicitive identity for this semi-ring.</p>
     *
     * <p>An element 1 is a multiplicitive identity if a * 1 = a for any a in the semi-ring.</p>
     *
     * @return True if this value is a multiplicitive identity for this semi-ring. Otherwise, false.
     */
    @Override
    public boolean isOne() {
        return equals(ONE);
    }


    /**
     * <p>Gets the additive identity for this semi-ring.</p>
     *
     * <p>An element 0 is an additive identity if a + 0 = a for any a in the semi-ring.</p>
     *
     * @return The additive identity for this semi-ring.
     */
    @Override
    public RealFloat64 getZero() {
        return ZERO;
    }


    /**
     * <p>Gets the multiplicitive identity for this semi-ring.</p>
     *
     * <p>An element 1 is a multiplicitive identity if a * 1 = a for any a in the semi-ring.</p>
     *
     * @return The multiplicitive identity for this semi-ring.
     */
    @Override
    public RealFloat64 getOne() {
        return ONE;
    }


    /**
     * Computes the quotient of two elements of this field.
     * @param b Second field element in quotient.
     * @return The quotient of this field element and {@code b}.
     */
    @Override
    public RealFloat64 div(RealFloat64 b) {
        return new RealFloat64(this.value / b.value);
    }


    /**
     * Sums an element of this field with a real number (associative and commutative).
     *
     * @param b Real element in sum.
     *
     * @return The sum of this element and {@code b}.
     */
    @Override
    public RealFloat64 add(double b) {
        return new RealFloat64(value + b);
    }


    /**
     * Computes difference of an element of this field and a real number.
     *
     * @param b Real value in difference.
     *
     * @return The difference of this ring element and {@code b}.
     */
    @Override
    public RealFloat64 sub(double b) {
        return new RealFloat64(value - b);
    }


    /**
     * Multiplies an element of this field with a real number (associative and commutative).
     *
     * @param b Real number in product.
     *
     * @return The product of this field element and {@code b}.
     */
    @Override
    public RealFloat64 mult(double b) {
        return new RealFloat64(value * b);
    }


    /**
     * Computes the quotient of an element of this field and a real number.
     *
     * @param b Real number in quotient.
     *
     * @return The quotient of this field element and {@code b}.
     */
    @Override
    public RealFloat64 div(double b) {
        return new RealFloat64(value / b);
    }


    /**
     * <p>Computes the multiplicitive inverse for an element of this field.</p>
     *
     * <p>An element x<sup>-1</sup> is a multaplicitive inverse for a filed element x if x<sup>-1</sup>*x = 1 where 1 is the
     * multiplicitive identity.</p>
     *
     * @return The multiplicitive inverse for this field element.
     */
    @Override
    public RealFloat64 multInv() {
        return new RealFloat64(1d/this.value);
    }


    /**
     * <p>Computes the addative inverse for an element of this field.</p>
     *
     * <p>An element -x is an addative inverse for a filed element x if -x + x = 0 where 0 is the addative identity..</p>
     *
     * @return The additive inverse for this field element.
     */
    @Override
    public RealFloat64 addInv() {
        return new RealFloat64(-this.value);
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
     * Computes the square root of this field element.
     *
     * @return The square root of this field element.
     */
    @Override
    public RealFloat64 sqrt() {
        return new RealFloat64((float) Math.sqrt(this.value));
    }


    /**
     * Evaluates the signum or sign function on a field element.
     *
     * @param a Value to evalute signum funciton on.
     * @return The output of the signum function evaluated on {@code a}.
     */
    public static RealFloat64 sgn(RealFloat64 a) {
        return new RealFloat64(Math.signum(a.value));
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
    public int compareTo(RealFloat64 b) {
        return Double.compare(this.value, b.value);
    }


    /**
     * Checks if this field element is finite in magnitude.
     *
     * @return True if this field element is finite in magnitude. False otherwise (i.e. infinite, NaN etc.).
     */
    @Override
    public boolean isFinite() {
        return Double.isFinite(this.value);
    }


    /**
     * Checks if this field element is infinite in magnitude.
     *
     * @return True if this field element is infinite in magnitude. False otherwise (i.e. finite, NaN, etc.).
     */
    @Override
    public boolean isInfinite() {
        return Double.isInfinite(this.value);
    }


    /**
     * Checks if this field element is NaN in magnitude.
     *
     * @return True if this field element is NaN in magnitude. False otherwise (i.e. finite, NaN, etc.).
     */
    @Override
    public boolean isNaN() {
        return Double.isNaN(this.value);
    }


    /**
     * Computes the product of all entires of specified array.
     * @param values Values to compute product of.
     * @return The product of all values in {@code values}.
     */
    public static RealFloat64 prod(RealFloat64... values) {
        if(values == null || values.length == 0) {
            throw new IllegalArgumentException("Values cannot be null or empty");
        }

        double prod = values[0].value;

        for(int i = 1, length=values.length; i < length; i++) {
            prod *= values[i].value;
        }

        return new RealFloat64(prod);
    }


    /**
     * Computes the sum of all entires of specified array.
     * @param values Values to compute product of.
     * @return The sum of all values in {@code values}.
     */
    public static RealFloat64 sum(RealFloat64... values) {
        if(values == null || values.length == 0) {
            throw new IllegalArgumentException("Values cannot be null or empty");
        }

        double prod = values[0].value;

        for(int i = 1, length=values.length; i < length; i++) {
            prod += values[i].value;
        }

        return new RealFloat64(prod);
    }


    /**
     * Wraps a primitive double array as a {@link RealFloat64} array.
     * @param arr Array to wrap.
     * @return A {@link RealFloat64} array containing the values of {@code arr}. If {@code arr==null} then {@code null} will be
     * returned.
     */
    public static RealFloat64[] wrapArray(double... arr) {
        if(arr == null) return null;

        RealFloat64[] wrapped = new RealFloat64[arr.length];
        for(int i=0, size=arr.length; i<size; i++)
            wrapped[i] = new RealFloat64(arr[i]);

        return wrapped;
    }


    /**
     * Wraps a {@link Double} array as a {@link RealFloat64} array.
     * @param arr Array to wrap.
     * @return A {@link RealFloat64} array containing the values of {@code arr}. If {@code arr==null} then {@code null} will be
     * returned.
     */
    public static RealFloat64[] wrapArray(Double[] arr) {
        if(arr == null) return null;

        RealFloat64[] wrapped = new RealFloat64[arr.length];
        for(int i=0, size=arr.length; i<size; i++)
            wrapped[i] = new RealFloat64(arr[i]);

        return wrapped;
    }


    /**
     * Checks if an object is equal to this Field element.
     * @param b Object to compare to this Field element.
     * @return True if the objects are the same or are both {@link RealFloat64}'s and have equal values.
     */
    @Override
    public boolean equals(Object b) {
        // Check for quick returns.
        if(this == b) return true;
        if(b == null) return false;
        if(b.getClass() != this.getClass()) return false;

        return this.value == ((RealFloat64) b).value;
    }


    /**
     * Converts this field element to a string representation.
     * @return A string representation of this field element.
     */
    public String toString() {
        return String.valueOf(this.value);
    }
}
