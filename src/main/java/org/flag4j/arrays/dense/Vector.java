/*
 * MIT License
 *
 * Copyright (c) 2024-2025. Jacob Watters
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

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.VectorMixin;
import org.flag4j.arrays.backend.primitive_arrays.AbstractDenseDoubleTensor;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.VectorNorms;
import org.flag4j.linalg.ops.common.complex.Complex128Ops;
import org.flag4j.linalg.ops.common.field_ops.FieldOps;
import org.flag4j.linalg.ops.dense.real.RealDenseVectorOps;
import org.flag4j.linalg.ops.dense.real_field_ops.RealFieldDenseElemDiv;
import org.flag4j.linalg.ops.dense.real_field_ops.RealFieldDenseElemMult;
import org.flag4j.linalg.ops.dense.real_field_ops.RealFieldDenseOps;
import org.flag4j.linalg.ops.dense_sparse.coo.real.RealDenseSparseVectorOps;
import org.flag4j.linalg.ops.dense_sparse.coo.real_complex.RealComplexDenseSparseVectorOps;
import org.flag4j.linalg.ops.dense_sparse.coo.real_field_ops.RealFieldDenseCooVectorOps;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.ArrayConversions;
import org.flag4j.util.StringUtils;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.flag4j.util.exceptions.TensorShapeException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * <p>A dense vector backed by a primitive double array.
 *
 * <p>Vectors are 1D tensors (i.e. rank 1 tensor).
 *
 * <p>Vectors have mutable data but are fixed in size.
 */
public class Vector extends AbstractDenseDoubleTensor<Vector>
        implements VectorMixin<Vector, Matrix, Matrix, Double> {
    private static final long serialVersionUID = 1L;

    /**
     * The size of this vector. That is, the number of data in this vector.
     */
    public final int size;


    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all data within the tensor.
     * If this tensor is sparse, this specifies only the non-zero data of the tensor.
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
        Arrays.fill(super.data, fillValue);
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
        Arrays.fill(super.data, fillValue);
        this.size = shape.get(0);
    }


    /**
     * Creates a vector with specified data.
     * @param entries Entries for this column vector.
     */
    public Vector(double... entries) {
        super(new Shape(entries.length), entries);
        size = shape.get(0);
    }


    /**
     * Creates a vector with specified data.
     * @param entries Entries for this column vector.
     */
    public Vector(int... entries) {
        super(new Shape(entries.length), new double[entries.length]);
        this.size = shape.get(0);

        for(int i=0; i<entries.length; i++) {
            super.data[i] = entries[i];
        }
    }


    /**
     * Creates a vector from another vector. This essentially copies the vector.
     * @param a Vector to make copy of.
     */
    public Vector(Vector a) {
        super(a.shape, a.data.clone());
        this.size = shape.get(0);
    }


    /**
     * Constructs a tensor of the same type as this tensor with the given the shape and data.
     *
     * @param shape Shape of the tensor to construct.
     * @param entries Entries of the tensor to construct.
     *
     * @return A tensor of the same type as this tensor with the given the shape and data.
     */
    @Override
    public Vector makeLikeTensor(Shape shape, double[] entries) {
        return new Vector(shape, entries);
    }


    /**
     * Flattens tensor to single dimension while preserving order of data.
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
     * @throws ArrayIndexOutOfBoundsException If the axis is not positive or larger than {@code this.{@link #getRank()}-1}.
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
                System.arraycopy(data, 0, tiled.data, i*tiled.numCols, size);
        } else {
            tiled = new Matrix(new Shape(size, n));

            for(int i=0, stop=tiled.numRows; i<stop; i++) // Fill each row of the tiled matrix with a single value from the vector.
                Arrays.fill(tiled.data, i*tiled.numCols, (i+1)*tiled.numCols, data[i]);
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
     * @throws IllegalArgumentException If the number of data in this vector is different from the number of data in
     *                                  the vector {@code b}.
     */
    @Override
    public Matrix stack(Vector b) {
        ValidateParameters.ensureEqualShape(shape, b.shape);
        Matrix stacked = new Matrix(2, size);

        // Copy data from each vector to the matrix.
        System.arraycopy(data, 0, stacked.data, 0, size);
        System.arraycopy(b.data, 0, stacked.data, size, b.size);

        return stacked;
    }


    /**
     * <p>
     * Stacks two vectors along specified axis.
     * 
     *
     * <p>
     * Stacking two vectors of length {@code n} along axis 0 stacks the vectors
     * as if they were row vectors resulting in a {@code 2&times;n} matrix.
     * 
     *
     * <p>
     * Stacking two vectors of length {@code n} along axis 1 stacks the vectors
     * as if they were column vectors resulting in a {@code n&times;2} matrix.
     * 
     *
     * @param b Vector to stack with this vector.
     * @param axis Axis along which to stack vectors. If {@code axis=0}, then vectors are stacked as if they are row
     * vectors. If {@code axis=1}, then vectors are stacked as if they are column vectors.
     *
     * @return The result of stacking this vector and the vector {@code b}.
     *
     * @throws IllegalArgumentException If the number of data in this vector is different from the number of
     *                                  data in the vector {@code b}.
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
                stackedEntries[i] = this.data[count];
                stackedEntries[i+1] = b.data[count++];
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
     * @throws IllegalArgumentException If the two vectors do not have the same number of data.
     */
    @Override
    public Matrix outer(Vector vector) {
        return RealDenseVectorOps.dispatchOuter(this, vector);
    }


    /**
     * Converts a vector to an equivalent matrix representing either a row or column vector.
     *
     * @param columVector Flag indicating whether to convert this vector to a matrix representing a row or column vector:
     * <p>If {@code true}, the vector will be converted to a matrix representing a column vector.
     * <p>If {@code false}, The vector will be converted to a matrix representing a row vector.
     *
     * @return A matrix equivalent to this vector.
     */
    @Override
    public Matrix toMatrix(boolean columVector) {
        if(columVector) {
            return new Matrix(this.data.length, 1, this.data.clone()); // Convert to column vector.
        } else {
            return new Matrix(1, this.data.length, this.data.clone()); // Convert to row vector.
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
        System.arraycopy(this.data, 0, joined.data, 0, this.size);
        System.arraycopy(b.data, 0, joined.data, this.size, b.size);

        return joined;
    }


    /**
     * Computes the inner product between two vectors.
     *
     * @param b Second vector in the inner product.
     *
     * @return The inner product between this vector and the vector {@code b}.
     *
     * @throws IllegalArgumentException If this vector and vector {@code b} do not have the same number of data.
     */
    @Override
    public Double inner(Vector b) {
        return RealDenseVectorOps.innerProduct(data, b.data);
    }


    /**
     * <p>Computes the dot product between two vectors.
     *
     * @param b Second vector in the dot product.
     *
     * @return The dot product between this vector and the vector {@code b}.
     *
     * @throws IllegalArgumentException If this vector and vector {@code b} do not have the same number of data.
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
        return VectorNorms.norm(data);
    }


    /**
     * Computes the p-norm of this vector.
     *
     * @param p {@code p} value in the p-norm.
     *
     * @return The Euclidean norm of this vector.
     */
    public double norm(int p) {
        return VectorNorms.norm(data, p);
    }


    /**
     * Computes a unit vector in the same direction as this vector.
     *
     * @return A unit vector with the same direction as this vector. If this vector is zeros, then an equivalently sized
     * zero vector will be returned.
     */
    public Vector normalize() {
        double norm = VectorNorms.norm(data);
        return norm==0 ? new Vector(size) : (Vector) this.div(norm);
    }


    /**
     * Computes the magnitude of this vector.
     *
     * @return The magnitude of this vector.
     */
    @Override
    public Double mag() {
        return VectorNorms.norm(data);
    }


    /**
     * Gets the element of this vector at the specified index.
     *
     * @param idx Index of the element to get within this vector.
     *
     * @return The element of this vector at index {@code idx}.
     */
    @Override
    public Double get(int idx) {
        ValidateParameters.validateTensorIndex(shape, idx);
        return data[idx];
    }


    /**
     * Computes the vector cross product between two vectors.
     *
     * @param b Second vector in the cross product.
     *
     * @return The result of the vector cross product between this vector and {@code b}.
     *
     * @throws IllegalArgumentException If either this vector or {@code b} do not have exactly 3 data.
     */
    public Vector cross(Vector b) {
        if(size != 3 || b.size != 3) {
            throw new LinearAlgebraException("Cross products can only be called vectors of size 3 but got sizes "
                    + size + " and " + b.size);
        }

        double[] entries = new double[3];

        entries[0] = data[1]*b.data[2] - data[2]*b.data[1];
        entries[1] = data[2]*b.data[0] - data[0]*b.data[2];
        entries[2] = data[0]*b.data[1] - data[1]*b.data[0];

        return new Vector(entries);
    }


    /**
     * Checks if a vector is parallel to this vector.
     *
     * @param b Vector to compare to this vector.
     *
     * @return {@code true} if the vector {@code b} is parallel to this vector and the same size; {@code false} otherwise.
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
                if(b.data[i]!=0) {
                    scale = this.data[i]/b.data[i];
                    break;
                }
            }

            // Ensure all data of b are the same scalar multiple of the data in this vector.
            for(int i=0, size=this.size; i<size; i++) {
                if(b.data[i]*scale != this.data[i]) {
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
     * @return {@code true} if the vector {@code b} is perpendicular to this vector and the same size; {@code false} otherwise.
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
     * @return The length, i.e. the number of data, in this vector.
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
        List<Double> nonZeroEntries = new ArrayList<>((int) (data.length*0.5));
        List<Integer> indices = new ArrayList<>((int) (data.length*0.5));

        // Fill data with non-zero values.
        for(int i=0; i<size; i++) {
            if(data[i] != 0.0) {
                nonZeroEntries.add(data[i]);
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
        return new CVector(ArrayConversions.toComplex128(data, null));
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
            data[i] *= b.data[i];
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
            data[i] += b.data[i];
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
            data[i] -= b.data[i];
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
            data[i] /= b.data[i];
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
            quotient[i] = this.data[i] / b.data[i];

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

        return shape.equals(src2.shape) && Arrays.equals(data, src2.data);
    }


    @Override
    public int hashCode() {
        int hash = 17;
        hash = 31*hash + shape.hashCode();
        hash = 31*hash + Arrays.hashCode(data);

        return hash;
    }


    /**
     * Adds a complex dense vector to this vector.
     * @param b Complex dense vector in the sum.
     * @return The sum of this vector and {@code b}.
     */
    public CVector add(CVector b) {
        Complex128[] dest = new Complex128[data.length];
        RealFieldDenseOps.add(b.shape, b.data, shape, data, dest);
        return new CVector(dest);
    }


    /**
     * Adds a real sparse vector to this vector.
     * @param b The real sparse vector in the sum.
     * @return The sum of this vector and {@code b}.
     */
    public Vector add(CooVector b) {
        return RealDenseSparseVectorOps.add(this, b);
    }


    /**
     * Adds a complex sparse vector to this vector.
     * @param b The complex sparse vector in the sum.
     * @return The sum of this vector and {@code b}.
     */
    public CVector add(CooCVector b) {
        return RealComplexDenseSparseVectorOps.add(this, b);
    }


    /**
     * Adds a complex-valued scalar value to each entry of this tensor. If the tensor is sparse, the scalar will only be added to the
     * non-zero data of the tensor.
     *
     * @param b Scalar value in sum.
     *
     * @return The sum of this tensor with the scalar {@code b}.
     */
    public CVector add(Complex128 b) {
        Complex128[] dest = new Complex128[data.length];
        RealFieldDenseOps.add(data, b, dest);
        return new CVector(dest);
    }


    /**
     * Subtracts a complex dense vector from this vector.
     * @param b Complex dense vector in the difference.
     * @return The difference of this vector and {@code b}.
     */
    public CVector sub(CVector b) {
        Complex128[] dest = new Complex128[data.length];
        RealFieldDenseOps.sub(shape, data, b.shape, b.data, dest);
        return new CVector(dest);
    }


    /**
     * Subtracts a real sparse vector from this vector.
     * @param b The real sparse vector in the difference.
     * @return The difference of this vector and {@code b}.
     */
    public Vector sub(CooVector b) {
        return RealDenseSparseVectorOps.sub(this, b);
    }


    /**
     * Subtracts a complex sparse vector from this vector.
     * @param b The complex sparse vector in the difference.
     * @return The difference of this vector and {@code b}.
     */
    public CVector sub(CooCVector b) {
        return RealComplexDenseSparseVectorOps.sub(this, b);
    }


    /**
     * Subtracts a complex-valued scalar from each entry of this vector.
     * @param b The scalar value in the difference.
     * @return The difference of this vector's data with the scalar value {@code b}.
     */
    public CVector sub(Complex128 b) {
        Complex128 bInv = b.addInv();
        Complex128[] sum = new Complex128[size];

        for(int i=0; i<size; i++)
            sum[i] = bInv.add(data[i]);

        return new CVector(sum);
    }


    /**
     * Multiplies this vector by a complex-valued scalar.
     * @param b Scalar to multiply this vector by.
     * @return The scalar product of this vector with {@code b}.
     */
    public CVector mult(Complex128 b) {
        Complex128[] dest = new Complex128[data.length];
        FieldOps.scalMult(data, b, dest);
        return new CVector(dest);
    }


    /**
     * Divides this vector by a complex-valued scalar.
     * @param b Scalar to divide this vector by.
     * @return The scalar quotient of this vector with {@code b}.
     */
    public CVector div(Complex128 b) {
        return new CVector(Complex128Ops.scalDiv(data, b));
    }


    /**
     * Computes the element-wise product of this vector and a complex dense vector.
     * @param b The complex dense vector in the element-wise product.
     * @return The element-wise product of this vector and {@code b}.
     */
    public CVector elemMult(CVector b) {
        Complex128[] dest = new Complex128[data.length];
        RealFieldDenseElemMult.dispatch(b.data, b.shape, data, shape, dest);
        return new CVector(dest);
    }


    /**
     * Computes the element-wise product of this vector and a real sparse vector.
     * @param b The real sparse vector in the element-wise product.
     * @return The element-wise product of this vector and {@code b}.
     */
    public CooVector elemMult(CooVector b) {
        return RealDenseSparseVectorOps.elemMult(this, b);
    }


    /**
     * Computes the element-wise product of this vector and a complex sparse vector.
     * @param b The complex sparse vector in the element-wise product.
     * @return The element-wise product of this vector and {@code b}.
     */
    public CooCVector elemMult(CooCVector b) {
        return (CooCVector) RealFieldDenseCooVectorOps.elemMult(this, b);
    }


    /**
     * Computes the element-wise quotient of this vector and a complex dense vector.
     * @param b The complex dense vector in the element-wise quotient.
     * @return The element-wise quotient of this vector and {@code b}.
     */
    public CVector div(CVector b) {
        Complex128[] dest = new Complex128[data.length];
        RealFieldDenseElemDiv.dispatch(shape, data, b.shape, b.data, dest);
        return new CVector(dest);
    }


    /**
     * Converts this vector to an equivalent tensor.
     * @return A tensor equivalent to this vector.
     */
    public Tensor toTensor() {
        return new Tensor(shape, data.clone());
    }


    /**
     * Converts this vector to a human-readable string format. To specify the maximum number of data to print, use
     * {@link PrintOptions#setMaxColumns(int)}.
     * @return A human-readable string representation of this vector.
     */
    public String toString() {
        StringBuilder result = new StringBuilder("shape: ").append(shape).append("\n");
        result.append("[");

        int stopIndex = Math.min(PrintOptions.getMaxColumns()-1, size-1);
        int width;
        String value;

        // Get data up until the stopping point.
        int padding = PrintOptions.getPadding();
        int precision = PrintOptions.getPrecision();
        boolean centering = PrintOptions.useCentering();

        for(int i = 0; i<stopIndex; i++) {
            value = StringUtils.ValueOfRound(data[i], precision);
            width = padding + value.length();
            value = centering ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        if(stopIndex < size-1) {
            width = padding + 3;
            value = "...";
            value = centering ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        // Get last entry now
        value = StringUtils.ValueOfRound(data[size-1], PrintOptions.getPrecision());
        width = padding + value.length();
        value = centering ? StringUtils.center(value, width) : value;
        result.append(String.format("%-" + width + "s", value));

        result.append("]");

        return result.toString();
    }
}
