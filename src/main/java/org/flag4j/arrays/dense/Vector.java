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

package org.flag4j.arrays.dense;

import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.VectorMixin;
import org.flag4j.arrays.backend.primitive.AbstractDenseDoubleTensor;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.VectorNorms;
import org.flag4j.linalg.operations.common.complex.Complex128Ops;
import org.flag4j.linalg.operations.common.field_ops.FieldOps;
import org.flag4j.linalg.operations.dense.real.RealDenseVectorOperations;
import org.flag4j.linalg.operations.dense.real_field_ops.RealFieldDenseElemDiv;
import org.flag4j.linalg.operations.dense.real_field_ops.RealFieldDenseElemMult;
import org.flag4j.linalg.operations.dense.real_field_ops.RealFieldDenseOperations;
import org.flag4j.linalg.operations.dense_sparse.coo.real.RealDenseSparseVectorOperations;
import org.flag4j.linalg.operations.dense_sparse.coo.real_field_ops.RealFieldDenseCooVectorOperations;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.StringUtils;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.flag4j.util.exceptions.TensorShapeException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * <p>A dense vector backed by a primitive double array.</p>
 *
 * <p>Vectors are 1D tensors (i.e. rank 1 tensor).</p>
 *
 * <p>Vectors have mutable entries but are fixed in size.</p>
 */
public class Vector extends AbstractDenseDoubleTensor<Vector>
        implements VectorMixin<Vector, Matrix, Matrix, Double> {

    /**
     * The size of this vector. That is, the number of entries in this vector.
     */
    public final int size;


    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all entries within the tensor.
     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     */
    public Vector(Shape shape, double[] entries) {
        super(shape, entries);
        ValidateParameters.ensureRank(shape, 1);
        size = shape.get(0);
    }


    /**
     * Creates a vector of specified size filled with zeros.
     * @param size Size of the vector.
     */
    public Vector(int size) {
        super(new Shape(size), new double[size]);
        this.size = shape.get(0);
    }


    /**
     * Creates a vector of specified size filled with a specified value.
     * @param size Size of the vector.
     * @param fillValue Value to fill vector with.
     */
    public Vector(int size, double fillValue) {
        super(new Shape(size), new double[size]);
        Arrays.fill(super.entries, fillValue);
        this.size = shape.get(0);
    }


    /**
     * Creates a vector of the specified shape filled with zeros.
     * @param shape Shape of this vector.
     * @throws IllegalArgumentException If the shapes is not rank 1.
     */
    public Vector(Shape shape) {
        super(shape, new double[shape.get(0)]);
        ValidateParameters.ensureRank(shape, 1);
        this.size = shape.get(0);
    }


    /**
     * Creates a vector of specified size filled with a specified value.
     * @param shape Shape of the vector.
     * @param fillValue Value to fill vector with.
     * @throws IllegalArgumentException If the shapes is not rank 1.
     */
    public Vector(Shape shape, double fillValue) {
        super(shape, new double[shape.get(0)]);
        ValidateParameters.ensureRank(shape, 1);
        Arrays.fill(super.entries, fillValue);
        this.size = shape.get(0);
    }


    /**
     * Creates a vector with specified entries.
     * @param entries Entries for this column vector.
     */
    public Vector(double... entries) {
        super(new Shape(entries.length), entries.clone());
        this.size = shape.get(0);
    }


    /**
     * Creates a vector with specified entries.
     * @param entries Entries for this column vector.
     */
    public Vector(int... entries) {
        super(new Shape(entries.length), new double[entries.length]);
        this.size = shape.get(0);

        for(int i=0; i<entries.length; i++) {
            super.entries[i] = entries[i];
        }
    }


    /**
     * Creates a vector from another vector. This essentially copies the vector.
     * @param a Vector to make copy of.
     */
    public Vector(Vector a) {
        super(a.shape, a.entries.clone());
        this.size = shape.get(0);
    }


    /**
     * Constructs a tensor of the same type as this tensor with the given the shape and entries.
     *
     * @param shape Shape of the tensor to construct.
     * @param entries Entries of the tensor to construct.
     *
     * @return A tensor of the same type as this tensor with the given the shape and entries.
     */
    @Override
    public Vector makeLikeTensor(Shape shape, double[] entries) {
        return new Vector(shape, entries);
    }


    /**
     * Flattens tensor to single dimension while preserving order of entries.
     *
     * @return The flattened tensor.
     *
     * @see #flatten(int)
     */
    @Override
    public Vector flatten() {
        return copy();
    }


    /**
     * Flattens a tensor along the specified axis.
     *
     * @param axis Axis along which to flatten tensor.
     *
     * @throws ArrayIndexOutOfBoundsException If the axis is not positive or larger than <code>this.{@link #getRank()}-1</code>.
     * @see #flatten()
     */
    @Override
    public Vector flatten(int axis) {
        ValidateParameters.ensureValidAxes(shape, axis);
        return copy();
    }


    /**
     * Repeats a vector {@code n} times along a certain axis to create a matrix.
     *
     * @param n Number of times to repeat vector.
     * @param axis Axis along which to repeat vector:
     * <ul>
     *     <li>If {@code axis=0}, then the vector will be treated as a row vector and stacked vertically {@code n} times.</li>
     *     <li>If {@code axis=1} then the vector will be treated as a column vector and stacked horizontally {@code n} times.</li>
     * </ul>
     *
     * @return A matrix whose rows/columns are this vector repeated.
     */
    @Override
    public Matrix repeat(int n, int axis) {
        ValidateParameters.ensureValidAxes(2, axis);
        ValidateParameters.ensureNonNegative(n);
        Matrix tiled;

        if(axis==0) {
            tiled = new Matrix(new Shape(n, size));

            for(int i=0, stop=tiled.numRows; i<stop; i++) // Set each row of the tiled matrix to be the vector values.
                System.arraycopy(entries, 0, tiled.entries, i*tiled.numCols, size);
        } else {
            tiled = new Matrix(new Shape(size, n));

            for(int i=0, stop=tiled.numRows; i<stop; i++) // Fill each row of the tiled matrix with a single value from the vector.
                Arrays.fill(tiled.entries, i*tiled.numCols, (i+1)*tiled.numCols, entries[i]);
        }

        return tiled;
    }


    /**
     * Stacks two vectors vertically as if they were row vectors to form a matrix with two rows.
     *
     * @param b Vector to stack below this vector.
     *
     * @return The result of stacking this vector and vector {@code b}.
     *
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of entries in
     *                                  the vector {@code b}.
     */
    @Override
    public Matrix stack(Vector b) {
        ValidateParameters.ensureEqualShape(shape, b.shape);
        Matrix stacked = new Matrix(2, size);

        // Copy entries from each vector to the matrix.
        System.arraycopy(entries, 0, stacked.entries, 0, size);
        System.arraycopy(b.entries, 0, stacked.entries, size, b.size);

        return stacked;
    }


    /**
     * <p>
     * Stacks two vectors along specified axis.
     * </p>
     *
     * <p>
     * Stacking two vectors of length {@code n} along axis 0 stacks the vectors
     * as if they were row vectors resulting in a {@code 2-by-n} matrix.
     * </p>
     *
     * <p>
     * Stacking two vectors of length {@code n} along axis 1 stacks the vectors
     * as if they were column vectors resulting in a {@code n-by-2} matrix.
     * </p>
     *
     * @param b Vector to stack with this vector.
     * @param axis Axis along which to stack vectors. If {@code axis=0}, then vectors are stacked as if they are row
     * vectors. If {@code axis=1}, then vectors are stacked as if they are column vectors.
     *
     * @return The result of stacking this vector and the vector {@code b}.
     *
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of
     *                                  entries in the vector {@code b}.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    @Override
    public Matrix stack(Vector b, int axis) {
        ValidateParameters.ensureAxis2D(axis);
        Matrix stacked;

        if(axis==0) {
            stacked = stack(b);
        } else {
            ValidateParameters.ensureArrayLengthsEq(this.size, b.size);
            double[] stackedEntries = new double[2*this.size];

            int count = 0;
            for(int i=0; i<stackedEntries.length; i+=2) {
                stackedEntries[i] = this.entries[count];
                stackedEntries[i+1] = b.entries[count++];
            }

            stacked = new Matrix(this.size, 2, stackedEntries);
        }

        return stacked;
    }


    /**
     * Computes the outer product of two vectors.
     *
     * @param vector Second vector in the outer product.
     *
     * @return The result of the vector outer product between this vector and {@code b}.
     *
     * @throws IllegalArgumentException If the two vectors do not have the same number of entries.
     */
    @Override
    public Matrix outer(Vector vector) {
        return RealDenseVectorOperations.dispatchOuter(this, vector);
    }


    /**
     * Converts a vector to an equivalent matrix representing either a row or column vector.
     *
     * @param columVector Flag indicating whether to convert this vector to a matrix representing a row or column vector:
     * <p>If {@code true}, the vector will be converted to a matrix representing a column vector.</p>
     * <p>If {@code false}, The vector will be converted to a matrix representing a row vector.</p>
     *
     * @return A matrix equivalent to this vector.
     */
    @Override
    public Matrix toMatrix(boolean columVector) {
        if(columVector) {
            return new Matrix(this.entries.length, 1, this.entries.clone()); // Convert to column vector.
        } else {
            return new Matrix(1, this.entries.length, this.entries.clone()); // Convert to row vector.
        }
    }


    /**
     * Joints specified vector with this vector. That is, creates a vector of length {@code this.length() + b.length()} containing
     * first the elements of this vector followed by the elements of {@code b}.
     *
     * @param b Vector to join with this vector.
     *
     * @return A vector resulting from joining the specified vector with this vector.
     */
    @Override
    public Vector join(Vector b) {
        Vector joined = new Vector(this.size+b.size);
        System.arraycopy(this.entries, 0, joined.entries, 0, this.size);
        System.arraycopy(b.entries, 0, joined.entries, this.size, b.size);

        return joined;
    }


    /**
     * Computes the inner product between two vectors.
     *
     * @param b Second vector in the inner product.
     *
     * @return The inner product between this vector and the vector {@code b}.
     *
     * @throws IllegalArgumentException If this vector and vector {@code b} do not have the same number of entries.
     */
    @Override
    public Double inner(Vector b) {
        return RealDenseVectorOperations.innerProduct(entries, b.entries);
    }


    /**
     * <p>Computes the dot product between two vectors.</p>
     *
     * @param b Second vector in the dot product.
     *
     * @return The dot product between this vector and the vector {@code b}.
     *
     * @throws IllegalArgumentException If this vector and vector {@code b} do not have the same number of entries.
     * @see #inner(Vector) 
     */
    @Override
    public Double dot(Vector b) {
        return inner(b);
    }


    /**
     * Computes the Euclidean norm of this vector.
     *
     * @return The Euclidean norm of this vector.
     */
    public double norm() {
        return VectorNorms.norm(entries);
    }


    /**
     * Computes the p-norm of this vector.
     *
     * @param p {@code p} value in the p-norm.
     *
     * @return The Euclidean norm of this vector.
     */
    public double norm(int p) {
        return VectorNorms.norm(entries, p);
    }


    /**
     * Computes a unit vector in the same direction as this vector.
     *
     * @return A unit vector with the same direction as this vector. If this vector is zeros, then an equivalently sized
     * zero vector will be returned.
     */
    public Vector normalize() {
        double norm = VectorNorms.norm(entries);
        return norm==0 ? new Vector(size) : (Vector) this.div(norm);
    }


    /**
     * Computes the magnitude of this vector.
     *
     * @return The magnitude of this vector.
     */
    @Override
    public Double mag() {
        return VectorNorms.norm(entries);
    }


    /**
     * Computes the vector cross product between two vectors.
     *
     * @param b Second vector in the cross product.
     *
     * @return The result of the vector cross product between this vector and {@code b}.
     *
     * @throws IllegalArgumentException If either this vector or {@code b} do not have exactly 3 entries.
     */
    public Vector cross(Vector b) {
        ValidateParameters.ensureEquals(3, b.size, this.size);
        double[] entries = new double[3];

        entries[0] = this.entries[1]*b.entries[2]-this.entries[2]*b.entries[1];
        entries[1] = this.entries[2]*b.entries[0]-this.entries[0]*b.entries[2];
        entries[2] = this.entries[0]*b.entries[1]-this.entries[1]*b.entries[0];

        return new Vector(entries);
    }


    /**
     * Checks if a vector is parallel to this vector.
     *
     * @param b Vector to compare to this vector.
     *
     * @return True if the vector {@code b} is parallel to this vector and the same size. Otherwise, returns false.
     *
     * @see #isPerp(Vector)
     */
    public boolean isParallel(Vector b) {
        if(this.size!=b.size) {
            return false;
        } else if(this.size==1) {
            return true;
        } else if(this.isZeros() || b.isZeros()) {
            return true; // Any vector is parallel to zero vector.
        } else {
            double scale = 0;

            // Find first non-zero entry of b to compute the scaling factor.
            for(int i=0, size=b.size; i<size; i++) {
                if(b.entries[i]!=0) {
                    scale = this.entries[i]/b.entries[i];
                    break;
                }
            }

            // Ensure all entries of b are the same scalar multiple of the entries in this vector.
            for(int i=0, size=this.size; i<size; i++) {
                if(b.entries[i]*scale != this.entries[i]) {
                    return false;
                }
            }
        }

        return true; // If we make it to here, the vectors must be parallel.
    }


    /**
     * Checks if a vector is perpendicular to this vector.
     *
     * @param b Vector to compare to this vector.
     *
     * @return True if the vector {@code b} is perpendicular to this vector and the same size. Otherwise, returns false.
     *
     * @see #isParallel(Vector)
     */
    public boolean isPerp(Vector b) {
        boolean result;

        if(this.size!=b.size) result = false;
        else result = this.inner(b)==0;

        return result;
    }


    /**
     * Gets the length of a vector.
     *
     * @return The length, i.e. the number of entries, in this vector.
     */
    @Override
    public int length() {
        return size;
    }


    /**
     * Computes the transpose of a tensor by exchanging {@code axis1} and {@code axis2}.
     *
     * @param axis1 First axis to exchange.
     * @param axis2 Second axis to exchange.
     *
     * @return The transpose of this tensor according to the specified axes.
     *
     * @throws IndexOutOfBoundsException If either {@code axis1} or {@code axis2} are out of bounds for the rank of this tensor.
     * @see #T()
     * @see #T(int...)
     */
    @Override
    public Vector T(int axis1, int axis2) {
        if(axis1 == axis2 && axis1 == 0) return copy();
        else throw new LinearAlgebraException(String.format("Cannot transpose axes [%d, %d] of tensor with rank 1.", axis1, axis2));
    }


    /**
     * Computes the transpose of this tensor. That is, permutes the axes of this tensor so that it matches
     * the permutation specified by {@code axes}.
     *
     * @param axes Permutation of tensor axis. If the tensor has rank {@code N}, then this must be an array of length
     * {@code N} which is a permutation of {@code {0, 1, 2, ..., N-1}}.
     *
     * @return The transpose of this tensor with its axes permuted by the {@code axes} array.
     *
     * @throws IndexOutOfBoundsException If any element of {@code axes} is out of bounds for the rank of this tensor.
     * @throws IllegalArgumentException  If {@code axes} is not a permutation of {@code {1, 2, 3, ... N-1}}.
     * @see #T(int, int)
     * @see #T()
     */
    @Override
    public Vector T(int... axes) {
        if(axes.length == 1 && axes[0] == 0) return copy();
        else throw new LinearAlgebraException(String.format("Cannot transpose axes %s of tensor with rank 1.", Arrays.toString(axes)));
    }


    /**
     * Converts this dense tensor to an equivalent sparse COO tensor.
     *
     * @return A sparse COO tensor equivalent to this dense tensor.
     */
    public CooVector toCoo() {
        // Estimate sparsity.
        List<Double> nonZeroEntries = new ArrayList<>((int) (entries.length*0.5));
        List<Integer> indices = new ArrayList<>((int) (entries.length*0.5));

        // Fill entries with non-zero values.
        for(int i=0; i<size; i++) {
            if(entries[i] != 0.0) {
                nonZeroEntries.add(entries[i]);
                indices.add(i);
            }
        }

        return new CooVector(size, nonZeroEntries, indices);
    }


    /**
     * Converts this real dense vector to an equivalent complex dense vector.
     * @return A complex dense vector equivalent to this vector.
     */
    public CVector toComplex() {
        return new CVector(ArrayUtils.wrapAsComplex128(entries, null));
    }


    /**
     * Computes the element-wise multiplication of two tensors and stores the result in this tensor.
     *
     * @param b Second tensor in the element-wise product.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public void elemMultEq(Vector b) {
        ValidateParameters.ensureEqualShape(shape, b.shape);

        for(int i=0; i<size; i++)
            entries[i] *= b.entries[i];
    }


    /**
     * Computes the element-wise sum between two tensors and stores the result in this tensor.
     *
     * @param b Second tensor in the element-wise sum.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public void addEq(Vector b) {
        ValidateParameters.ensureEqualShape(shape, b.shape);

        for(int i=0; i<size; i++)
            entries[i] += b.entries[i];
    }


    /**
     * Computes the element-wise difference between two tensors and stores the result in this tensor.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public void subEq(Vector b) {
        ValidateParameters.ensureEqualShape(shape, b.shape);

        for(int i=0; i<size; i++)
            entries[i] -= b.entries[i];
    }


    /**
     * Computes the element-wise division between two tensors and stores the result in this tensor.
     *
     * @param b The denominator tensor in the element-wise quotient.
     *
     * @throws TensorShapeException If this tensor and {@code b}'s shape are not equal.
     */
    @Override
    public void divEq(Vector b) {
        ValidateParameters.ensureEqualShape(shape, b.shape);

        for(int i=0; i<size; i++)
            entries[i] /= b.entries[i];
    }


    /**
     * Computes the element-wise division between two tensors.
     *
     * @param b The denominator tensor in the element-wise quotient.
     *
     * @return The element-wise quotient of this tensor and {@code b}.
     *
     * @throws TensorShapeException If this tensor and {@code b}'s shape are not equal.
     */
    @Override
    public Vector div(Vector b) {
        ValidateParameters.ensureEqualShape(shape, b.shape);
        double[] quotient = new double[size];

        for(int i=0; i<size; i++)
            quotient[i] = this.entries[i] / b.entries[i];

        return new Vector(quotient);
    }


    /**
     * Checks if an object is equal to this vector object.
     * @param object Object to check equality with this vector.
     * @return True if the two vectors have the same shape, are numerically equivalent, and are of type {@link Vector}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        Vector src2 = (Vector) object;

        return shape.equals(src2.shape) && Arrays.equals(entries, src2.entries);
    }


    @Override
    public int hashCode() {
        int hash = 17;
        hash = 31*hash + shape.hashCode();
        hash = 31*hash + Arrays.hashCode(entries);

        return hash;
    }


    /**
     * Adds a complex dense vector to this vector.
     * @param b Complex dense vector in the sum.
     * @return The sum of this vector and {@code b}.
     */
    public CVector add(CVector b) {
        return new CVector(RealFieldDenseOperations.add(b.entries, b.shape, entries, shape));
    }


    /**
     * Adds a real sparse vector to this vector.
     * @param b The real sparse vector in the sum.
     * @return The sum of this vector and {@code b}.
     */
    public Vector add(CooVector b) {
        return RealDenseSparseVectorOperations.add(this, b);
    }


    /**
     * Adds a complex sparse vector to this vector.
     * @param b The complex sparse vector in the sum.
     * @return The sum of this vector and {@code b}.
     */
    public CVector add(CooCVector b) {
//        return RealComplexDenseSparseVectorOperations.add(this, b);
        return null;
    }


    /**
     * Adds a complex-valued scalar to each entry of this vector.
     * @param b The scalar value in the sum.
     * @return The sum of this vector's entries with the scalar value {@code b}.
     */
    public CVector add(Complex128 b) {
        Complex128[] sum = new Complex128[size];

        for(int i=0; i<size; i++)
            sum[i] = b.add(entries[i]);

        return new CVector(sum);
    }


    /**
     * Subtracts a complex dense vector from this vector.
     * @param b Complex dense vector in the difference.
     * @return The difference of this vector and {@code b}.
     */
    public CVector sub(CVector b) {
        return new CVector(RealFieldDenseOperations.sub(entries, shape, b.entries, b.shape));
    }


    /**
     * Subtracts a real sparse vector from this vector.
     * @param b The real sparse vector in the difference.
     * @return The difference of this vector and {@code b}.
     */
    public Vector sub(CooVector b) {
        return RealDenseSparseVectorOperations.sub(this, b);
    }


    /**
     * Subtracts a complex sparse vector from this vector.
     * @param b The complex sparse vector in the difference.
     * @return The difference of this vector and {@code b}.
     */
    public CVector sub(CooCVector b) {
//        return RealComplexDenseSparseVectorOperations.sub(this, b);
        return null;
    }


    /**
     * Subtracts a complex-valued scalar from each entry of this vector.
     * @param b The scalar value in the difference.
     * @return The difference of this vector's entries with the scalar value {@code b}.
     */
    public CVector sub(Complex128 b) {
        Complex128 bInv = b.addInv();
        Complex128[] sum = new Complex128[size];

        for(int i=0; i<size; i++)
            sum[i] = bInv.add(entries[i]);

        return new CVector(sum);
    }


    /**
     * Multiplies this vector by a complex-valued scalar.
     * @param b Scalar to multiply this vector by.
     * @return The scalar product of this vector with {@code b}.
     */
    public CVector mult(Complex128 b) {
        return new CVector(FieldOps.scalMult(entries, b, null));
    }


    /**
     * Divides this vector by a complex-valued scalar.
     * @param b Scalar to divide this vector by.
     * @return The scalar quotient of this vector with {@code b}.
     */
    public CVector div(Complex128 b) {
        return new CVector(Complex128Ops.scalDiv(entries, b));
    }


    /**
     * Computes the element-wise product of this vector and a complex dense vector.
     * @param b The complex dense vector in the element-wise product.
     * @return The element-wise product of this vector and {@code b}.
     */
    public CVector elemMult(CVector b) {
        return new CVector(RealFieldDenseElemMult.dispatch(b.entries, b.shape, this.entries, this.shape));
    }


    /**
     * Computes the element-wise product of this vector and a real sparse vector.
     * @param b The real sparse vector in the element-wise product.
     * @return The element-wise product of this vector and {@code b}.
     */
    public CooVector elemMult(CooVector b) {
        return RealDenseSparseVectorOperations.elemMult(this, b);
    }


    /**
     * Computes the element-wise product of this vector and a complex sparse vector.
     * @param b The complex sparse vector in the element-wise product.
     * @return The element-wise product of this vector and {@code b}.
     */
    public CooCVector elemMult(CooCVector b) {
        return (CooCVector) RealFieldDenseCooVectorOperations.elemMult(this, b);
    }


    /**
     * Computes the element-wise quotient of this vector and a complex dense vector.
     * @param b The complex dense vector in the element-wise quotient.
     * @return The element-wise quotient of this vector and {@code b}.
     */
    public CVector div(CVector b) {
        return new CVector(RealFieldDenseElemDiv.dispatch(this.shape, this.entries, b.shape, b.entries));
    }


    /**
     * Converts this vector to an equivalent tensor.
     * @return A tensor equivalent to this vector.
     */
    public Tensor toTensor() {
        return new Tensor(shape, entries.clone());
    }


    /**
     * Converts this vector to a human-readable string format. To specify the maximum number of entries to print, use
     * {@link PrintOptions#setMaxColumns(int)}.
     * @return A human-readable string representation of this vector.
     */
    public String toString() {
        StringBuilder result = new StringBuilder("shape: ").append(shape).append("\n");
        result.append("[");

        int stopIndex = Math.min(PrintOptions.getMaxColumns()-1, size-1);
        int width;
        String value;

        // Get entries up until the stopping point.
        for(int i=0; i<stopIndex; i++) {
            value = StringUtils.ValueOfRound(entries[i], PrintOptions.getPrecision());
            width = PrintOptions.getPadding() + value.length();
            value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        if(stopIndex < size-1) {
            width = PrintOptions.getPadding() + 3;
            value = "...";
            value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        // Get last entry now
        value = StringUtils.ValueOfRound(entries[size-1], PrintOptions.getPrecision());
        width = PrintOptions.getPadding() + value.length();
        value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
        result.append(String.format("%-" + width + "s", value));

        result.append("]");

        return result.toString();
    }
}
