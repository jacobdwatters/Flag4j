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

/**
 * <p>A real number backed by a 32-bit floating point number. Immutable</p>
 *
 * <p>This class wraps the primative float type.</p>
 */
public class RealFloat32 implements Field<RealFloat32> {

    /**
     * Numerical value of field element.
     */
    private final float value;


    /**
     * Constructs a real 32-bit floating point number.
     * @param value Value of the 32-bit floating point number.
     */
    public RealFloat32(float value) {
        this.value = value;
    }



    /**
     * Constructs a real 32-bit floating point number by casting a real 64-bit floating point.
     * @param value Value of the 32-bit floating point number.
     */
    public RealFloat32(RealFloat64 value) {
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
    public RealFloat32 add(RealFloat32 b) {
        return new RealFloat32(this.value + b.value);
    }


    /**
     * Computes difference of two elements of this field.
     *
     * @param b Second field element in difference.
     *
     * @return The difference of this field element and {@code b}.
     */
    @Override
    public RealFloat32 sub(RealFloat32 b) {
        return new RealFloat32(this.value - b.value);
    }


    /**
     * Multiplies two elements of this field (associative and commutative).
     *
     * @param b Second field element in product.
     *
     * @return The product of this field element and {@code b}.
     */
    @Override
    public RealFloat32 mult(RealFloat32 b) {
        return new RealFloat32(this.value * b.value);
    }


    /**
     * Computes the quotient of two elements of this field.
     * @param b Second field element in quotient.
     * @return The quotient of this field element and {@code b}.
     */
    @Override
    public RealFloat32 div(RealFloat32 b) {
        return new RealFloat32(this.value / b.value);
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
    public RealFloat32 multInv() {
        return new RealFloat32(1f/this.value);
    }


    /**
     * <p>Computes the addative inverse for an element of this field.</p>
     *
     * <p>An element -x is an addative inverse for a filed element x if -x + x = 0 where 0 is the addative identity..</p>
     *
     * @return The additive inverse for this field element.
     */
    @Override
    public RealFloat32 addInv() {
        return new RealFloat32(-this.value);
    }


    /**
     * Computes the conjugate of this field element.
     *
     * @return The conjugate of this field element.
     */
    @Override
    public RealFloat32 conj() {
        return this;
    }


    /**
     * Computes the magnitude of this field element.
     *
     * @return The magniitude of this field element.
     */
    @Override
    public double mag() {
        return this.value;
    }


    /**
     * Computes the square root of this field element.
     *
     * @return The square root of this field element.
     */
    @Override
    public RealFloat32 sqrt() {
        return new RealFloat32((float) Math.sqrt(this.value));
    }


    /**
     * Evaluates the signum or sign function on a field element.
     *
     * @param a Value to evalute signum funciton on.
     * @return The output of the signum function evaluated on {@code a}.
     */
    public static RealFloat32 sgn(RealFloat32 a) {
        return new RealFloat32(Math.signum(a.value));
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
    public int compareTo(RealFloat32 b) {
        return Float.compare(this.value, b.value);
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
     * Computes the product of all entires of specified array.
     * @param values Values to compute product of.
     * @return The product of all values in {@code values}.
     */
    public static RealFloat32 prod(RealFloat32... values) {
        if(values == null || values.length == 0) {
            throw new IllegalArgumentException("Values cannot be null or empty");
        }

        float prod = values[0].value;

        for(int i = 1, length=values.length; i < length; i++) {
            prod *= values[i].value;
        }

        return new RealFloat32(prod);
    }


    /**
     * Computes the sum of all entires of specified array.
     * @param values Values to compute product of.
     * @return The sum of all values in {@code values}.
     */
    public static RealFloat32 sum(RealFloat32... values) {
        if(values == null || values.length == 0) {
            throw new IllegalArgumentException("Values cannot be null or empty");
        }

        float prod = values[0].value;

        for(int i = 1, length=values.length; i < length; i++) {
            prod += values[i].value;
        }

        return new RealFloat32(prod);
    }


    /**
     * Checks if an object is equal to this Field element.
     * @param b Object to compare to this Field element.
     * @return True if the objects are the same or are both {@link RealFloat32}'s and have equal values.
     */
    public boolean equls(Object b) {
        // Check for quick returns.
        if(this == b) return true;
        if(b == null) return false;
        if(b.getClass() != this.getClass()) return false;

        return this.value == ((RealFloat32) b).value;
    }


    /**
     * Converts this field element to a string representation.
     * @return A string representation of this field element.
     */
    public String toString() {
        return String.valueOf(this.value);
    }
}
