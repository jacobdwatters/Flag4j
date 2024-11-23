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

package org.flag4j.arrays.backend.field;

import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.backend.ring.TensorOverRing;
import org.flag4j.arrays.dense.Tensor;
import org.flag4j.linalg.VectorNorms;
import org.flag4j.linalg.operations.common.field_ops.FieldOps;
import org.flag4j.linalg.operations.common.ring_ops.CompareRing;
import org.flag4j.linalg.operations.common.ring_ops.RingOps;
import org.flag4j.linalg.operations.common.semiring_ops.AggregateSemiring;
import org.flag4j.linalg.operations.common.semiring_ops.CompareSemiring;
import org.flag4j.linalg.operations.common.semiring_ops.SemiRingOperations;
import org.flag4j.linalg.operations.common.semiring_ops.SemiRingProperties;

/**
 * <p>This interface provides default functionality for all tensors whose entries are elements of a
 * {@link org.flag4j.algebraic_structures.fields.Field}. This includes both sparse and dense tensors.</p>
 *
 * <p>The default methods in this interface can be overridden if desired, but it is generally recommended to use them as is.</p>
 * @param <T> Type of this tensor.
 * @param <U> Dense equivalent of this tensor. If this tensor is dense, this should be the same type as {@code T}
 * @param <V> Type of an element of this tensor. Satisfies {@link org.flag4j.algebraic_structures.fields.Field field} axioms.
 */
public interface FieldTensorMixin<T extends FieldTensorMixin<T, U, V>,
        U extends FieldTensorMixin<U, U, V>, V extends Field<V>>
        extends TensorOverField<T, U, Field<V>[], V> {

    /**
     * Subtracts a scalar value from each entry of this tensor.
     *
     * @param b Scalar value in difference.
     *
     * @return The difference of this tensor and the scalar {@code b}.
     */
    @Override
    default T sub(V b) {
        Field<V>[] entries = getEntries();
        Field<V>[] diff = new Field[entries.length];
        RingOps.sub(entries, b, diff);
        return makeLikeTensor(getShape(), diff);
    }


    /**
     * Subtracts a scalar value from each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar value in difference.
     */
    @Override
    default void subEq(V b) {
        Field<V>[] entries = getEntries();
        RingOps.sub(entries, b, entries);
    }


    /**
     * Computes the element-wise absolute value of this tensor.
     *
     * @return The element-wise absolute value of this tensor.
     */
    @Override
    default TensorOverRing abs() {
        Field<V>[] entries = getEntries();
        double[] abs = new double[entries.length];
        RingOps.abs(entries, abs);
        return new Tensor(getShape(), abs);
    }


    /**
     * Computes the element-wise conjugation of this tensor.
     *
     * @return The element-wise conjugation of this tensor.
     */
    @Override
    default T conj() {
        Field<V>[] entries = getEntries();
        Field<V>[] conj = new Field[entries.length];
        RingOps.conj(entries, conj);
        return makeLikeTensor(getShape(), conj);
    }


    /**
     * Computes the element-wise reciprocals of this tensor.
     * @return The element-wise reciprocals of this tensor.
     */
    @Override
    default T recip() {
        Field<V>[] entries = getEntries();
        Field<V>[] recip = new Field[entries.length];
        FieldOps.recip(entries, recip);
        return makeLikeTensor(getShape(), recip);
    }


    /**
     * Finds the minimum value in this tensor. If this tensor is complex, then this method finds the smallest value in magnitude.
     *
     * @return The minimum value (smallest in magnitude for a complex valued tensor) in this tensor.
     */
    default V min() {
        return (V) CompareSemiring.min(getEntries());
    }


    /**
     * Finds the maximum value in this tensor. If this tensor is complex, then this method finds the largest value in magnitude.
     *
     * @return The maximum value (largest in magnitude for a complex valued tensor) in this tensor.
     */
    default V max() {
        return (V) CompareSemiring.max(getEntries());
    }


    /**
     * Finds the indices of the minimum value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    default int[] argmin() {
        return getShape().getNdIndices(CompareSemiring.argmin(getEntries()));
    }


    /**
     * Finds the indices of the maximum value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    default int[] argmax() {
        return getShape().getNdIndices(CompareSemiring.argmax(getEntries()));
    }


    /**
     * Finds the indices of the minimum absolute value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    default int[] argminAbs() {
        return getShape().getNdIndices(CompareRing.argminAbs(getEntries()));
    }


    /**
     * Finds the indices of the maximum absolute value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    default int[] argmaxAbs() {
        return getShape().getNdIndices(CompareRing.argmaxAbs(getEntries()));
    }


    /**
     * Finds the minimum value, in absolute value, in this tensor.
     *
     * @return The minimum value, in absolute value, in this tensor.
     */
    @Override
    default double minAbs() {
        return CompareRing.minAbs(getEntries());
    }


    /**
     * Finds the maximum absolute value in this tensor.
     *
     * @return The maximum absolute value in this tensor.
     */
    @Override
    default double maxAbs() {
        return CompareRing.maxAbs(getEntries());
    }


    /**
     * Adds a scalar value to each entry of this tensor. If the tensor is sparse, the scalar will only be added to the non-zero
     * entries of the tensor.
     *
     * @param b Scalar field value in sum.
     *
     * @return The sum of this tensor with the scalar {@code b}.
     */
    @Override
    default T add(V b) {
        return makeLikeTensor(getShape(), (V[]) SemiRingOperations.add(getEntries(), null, b));
    }


    /**
     * Adds a scalar value to each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar field value in sum.
     */
    @Override
    default void addEq(V b) {
        Field<V>[] entries = getEntries();
        SemiRingOperations.add(entries, entries, b);
    }


    /**
     * Multiplies a scalar value to each entry of this tensor.
     *
     * @param b Scalar value in product.
     *
     * @return The product of this tensor with {@code b}.
     */
    @Override
    default T mult(V b) {
        Field<V>[] entries = getEntries();
        return makeLikeTensor(getShape(), (V[]) SemiRingOperations.scalMult(getEntries(), null, b));
    }


    /**
     * Multiplies a scalar value to each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar value in product.
     */
    @Override
    default void multEq(V b) {
        Field<V>[] entries = getEntries();
        SemiRingOperations.scalMult(entries, entries, b);
    }


    /**
     * Checks if this tensor only contains zeros.
     *
     * @return True if this tensor only contains zeros. Otherwise, returns false.
     */
    @Override
    default boolean isZeros() {
        return SemiRingProperties.isZeros(getEntries());
    }

    /**
     * Checks if this tensor only contains ones. If this tensor is sparse, only the non-zero entries are considered.
     *
     * @return True if this tensor only contains ones. Otherwise, returns false.
     */
    @Override
    default boolean isOnes() {
        return SemiRingProperties.isOnes(getEntries());
    }


    /**
     * Computes the sum of all values in this tensor.
     *
     * @return The sum of all values in this tensor.
     */
    @Override
    default V sum() {
        return AggregateSemiring.sum(getEntries());
    }


    /**
     * Computes the product of all values in this tensor (or non-zero values if this tensor is sparse).
     *
     * @return The product of all values (or non-zero values if sparse) in this tensor.
     */
    @Override
    default V prod() {
        return AggregateSemiring.prod(getEntries());
    }

    /**
     * Adds a primitive scalar value to each entry of this tensor. If the tensor is sparse, the scalar will only be added to the
     * non-zero
     * entries of the tensor.
     *
     * @param b Scalar field value in sum.
     *
     * @return The sum of this tensor with the scalar {@code b}.
     */
    @Override
    default T add(double b) {
        Field<V>[] entries = getEntries();
        Field<V>[] dest = new Field[entries.length];
        FieldOps.add(entries, b, dest);
        return makeLikeTensor(getShape(), entries);
    }

    /**
     * Adds a primitive scalar value to each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar field value in sum.
     */
    @Override
    default void addEq(double b) {
        Field<V>[] entries = getEntries();
        FieldOps.add(entries, b, entries);
    }

    /**
     * Multiplies a primitive scalar value to each entry of this tensor.
     *
     * @param b Scalar value in product.
     *
     * @return The product of this tensor with {@code b}.
     */
    @Override
    default T mult(double b) {
        Field<V>[] entries = getEntries();
        Field<V>[] dest = new Field[entries.length];
        FieldOps.mult(entries, b, dest);
        return makeLikeTensor(getShape(), entries);
    }

    /**
     * Multiplies a primitive scalar value to each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar value in product.
     */
    @Override
    default void multEq(double b) {
        Field<V>[] entries = getEntries();
        FieldOps.mult(entries, b, entries);
    }

    /**
     * Subtracts a primitive scalar value from each entry of this tensor.
     *
     * @param b Scalar value in difference.
     *
     * @return The difference of this tensor and the scalar {@code b}.
     */
    @Override
    default T sub(double b) {
        Field<V>[] entries = getEntries();
        Field<V>[] dest = new Field[entries.length];
        FieldOps.sub(entries, b, dest);
        return makeLikeTensor(getShape(), entries);
    }

    /**
     * Subtracts a scalar primitive value from each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar value in difference.
     */
    @Override
    default void subEq(double b) {
        Field<V>[] entries = getEntries();
        FieldOps.sub(entries, b, entries);
    }

    /**
     * Divides each element of this tensor by a scalar value.
     *
     * @param b Scalar value in quotient.
     *
     * @return The element-wise quotient of this tensor and the scalar {@code b}.
     *
     * @see #divEq(Field)
     * @see #div(double)
     * @see #divEq(double)
     */
    @Override
    default T div(V b) {
        Field<V>[] entries = getEntries();
        Field<V>[] dest = new Field[entries.length];
        FieldOps.div(entries, b, dest);
        return makeLikeTensor(getShape(), entries);
    }

    /**
     * Divides each element of this tensor by a scalar value and stores the result in this tensor.
     *
     * @param b Scalar value in quotient.
     *
     * @see #divEq(double)
     * @see #div(Field)
     * @see #divEq(double)
     */
    @Override
    default void divEq(V b) {
        Field<V>[] entries = getEntries();
        FieldOps.div(entries, b, entries);
    }

    /**
     * Divides each element of this tensor by a primitive scalar value.
     *
     * @param b Scalar value in quotient.
     *
     * @return The element-wise quotient of this tensor and the scalar {@code b}.
     *
     * @see #divEq(Field)
     * @see #div(Field)
     * @see #divEq(double)
     */
    @Override
    default T div(double b) {
        Field<V>[] entries = getEntries();
        Field<V>[] dest = new Field[entries.length];
        FieldOps.div(entries, b, dest);
        return makeLikeTensor(getShape(), entries);
    }

    /**
     * Divides each element of this tensor by a primitive scalar value and stores the result in this tensor.
     *
     * @param b Scalar value in quotient.
     *
     * @see #div(Field)
     * @see #divEq(Field)
     * @see #div(double)
     */
    @Override
    default void divEq(double b) {
        Field<V>[] entries = getEntries();
        FieldOps.div(entries, b, entries);
    }


    /**
     * Computes the Euclidean norm of this vector.
     *
     * @return The Euclidean norm of this vector.
     */
    default double norm() {
        return VectorNorms.norm(getEntries());
    }


    /**
     * Computes the p-norm of this vector.
     *
     * @param p {@code p} value in the p-norm.
     *
     * @return The Euclidean norm of this vector.
     */
    default double norm(int p) {
        return VectorNorms.norm(getEntries(), p);
    }
}
