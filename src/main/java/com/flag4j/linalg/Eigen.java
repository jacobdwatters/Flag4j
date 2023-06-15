/*
 * MIT License
 *
 * Copyright (c) 2022-2023 Jacob Watters
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

package com.flag4j.linalg;

import com.flag4j.CMatrix;
import com.flag4j.CVector;
import com.flag4j.Matrix;
import com.flag4j.Vector;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.linalg.decompositions.ComplexSchurDecomposition;
import com.flag4j.linalg.decompositions.RealSchurDecomposition;
import com.flag4j.linalg.decompositions.SchurDecomposition;
import com.flag4j.linalg.solvers.ComplexBackSolver;
import com.flag4j.linalg.solvers.RealBackSolver;

/**
 * This class provides several methods useful for computing eigen values, eigen vectors, as well as singular values and
 * singular vectors.
 */
public class Eigen {

    /**
     * Computes the eigenvalues of a 2x2 matrix explicitly.
     * @param src The 2x2 matrix to compute the eigenvalues of.
     * @return A complex vector containing the eigenvalues of the 2x2 {@code src} matrix.
     */
    public static CVector get2x2EigenValues(Matrix src) {
        // TODO: While theoretically correct, there are some numerical issues here.
        CVector lambda = new CVector(2);

        // Get the four entries from lower right 2x2 sub-matrix.
        double a = src.entries[0];
        double b = src.entries[1];
        double c = src.entries[2];
        double d = src.entries[3];

        double det = a*d - b*c; // 2x2 determinant.
        double htr = (a+d)/2; // Half of the 2x2 trace.

        // 2x2 block eigenvalues.
        lambda.entries[0] = CNumber.sqrt(CNumber.pow(htr, 2).sub(det)).add(htr);
        lambda.entries[1] = CNumber.sqrt(CNumber.pow(htr, 2).sub(det)).add(htr);

        return lambda;
    }


    /**
     * Computes the eigenvalues of a 2x2 matrix explicitly.
     * @param src The 2x2 matrix to compute the eigenvalues of.
     * @return A complex vector containing the eigenvalues of the 2x2 {@code src} matrix.
     */
    public static CVector get2x2EigenValues(CMatrix src) {
        // TODO: While theoretically correct, there are some numerical issues here.
        CVector lambda = new CVector(2);

        // Get the four entries from lower right 2x2 sub-matrix.
        CNumber a = src.entries[0];
        CNumber b = src.entries[1];
        CNumber c = src.entries[2];
        CNumber d = src.entries[3];

        CNumber det = a.mult(d).sub(b.mult(c)); // 2x2 determinant.
        CNumber htr = a.add(d).div(2); // Half of the 2x2 trace.

        // 2x2 block eigenvalues.
        lambda.entries[0] = htr.add(CNumber.sqrt(CNumber.pow(htr, 2).sub(det)));
        lambda.entries[1] = htr.sub(CNumber.sqrt(CNumber.pow(htr, 2).sub(det)));

        return lambda;
    }


    /**
     * Computes the eigenvalues for the lower right 2x2 block matrix of a larger matrix.
     * @param src Source matrix to compute eigenvalues of lower right 2x2 block.
     * @return A vector of length 2 containing the eigenvalues of the lower right 2x2 block of {@code src}.
     */
    public static CVector get2x2LowerLeftBlockEigenValues(CMatrix src) {
        // TODO: While theoretically correct, there are some numerical
        //  issues here.
        CVector lambdas = new CVector(2);
        int n = src.numRows-1;

        // Get the four entries from lower right 2x2 sub-matrix.
        CNumber a = src.entries[(n-1)*(src.numCols + 1)];
        CNumber b = src.entries[(n-1)*src.numCols + n];
        CNumber c = src.entries[n*(src.numCols + 1) - 1];
        CNumber d = src.entries[n*(src.numCols + 1)];

        CNumber det = a.mult(d).sub(b.mult(c)); // 2x2 determinant.
        CNumber htr = a.add(d).div(2); // Half of the 2x2 trace.

        // 2x2 block eigenvalues.
        lambdas.entries[0] = htr.add(CNumber.sqrt(CNumber.pow(htr, 2).sub(det)));
        lambdas.entries[1] = htr.sub(CNumber.sqrt(CNumber.pow(htr, 2).sub(det)));

        return lambdas;
    }


    /**
     * Computes the eigenvalues of a square real dense matrix.
     * @param src The matrix to compute the eigenvalues of.
     * @return The eigenvalues of the {@code src} matrix.
     */
    public static CVector getEigenValues(Matrix src) {
        CVector lambdas = new CVector(src.numRows);

        SchurDecomposition<Matrix, Vector> schur = new RealSchurDecomposition(false).decompose(src);
        CMatrix T = schur.getT();

        // Extract diagonal of T.
        for(int i=0; i<T.numRows; i++) {
            lambdas.entries[i] = T.entries[i*(T.numCols + 1)];
        }

        return lambdas;
    }


    /**
     * Computes the eigenvalues of a square complex dense matrix.
     * @param src The matrix to compute the eigenvalues of.
     * @return The eigenvalues of the {@code src} matrix.
     */
    public static CVector getEigenValues(CMatrix src) {
        CVector lambdas = new CVector(src.numRows);

        SchurDecomposition<CMatrix, CVector> schur = new ComplexSchurDecomposition(false).decompose(src);
        CMatrix T = schur.getT();

        // Extract diagonal of T.
        for(int i=0; i<T.numRows; i++) {
            lambdas.entries[i] = T.entries[i*(T.numCols + 1)];
        }

        return lambdas;
    }


    /**
     * Computes the eigenvectors of a square real dense matrix.
     * @param src The matrix to compute the eigenvectors of.
     * @return A matrix containing the eigenvectors of {@code src} as its columns.
     */
    public static CMatrix getEigenVectors(Matrix src) {
        SchurDecomposition<Matrix, Vector> schur = new RealSchurDecomposition(true).decompose(src);
        CMatrix U = schur.getU();

        if(src.isSymmetric()) {
            // Then the columns of U are the complete orthonormal set of eigenvectors of the src matrix.
            return U;
        } else {
            // For a non-symmetric matrix, only the first column of U will be an eigenvector of the src matrix.
            CMatrix Q = getEigenVectorsTriu(schur.getT()); // Compute the eigenvectors of T.
            return U.mult(Q); // Convert the eigenvectors of T to the eigenvectors of the src matrix.
        }
    }


    /**
     * Computes the eigenvectors of a square complex dense matrix.
     * @param src The matrix to compute the eigenvectors of.
     * @return A matrix containing the eigenvectors of {@code src} as its columns.
     */
    public static CMatrix getEigenVectors(CMatrix src) {
        SchurDecomposition<CMatrix, CVector> schur = new ComplexSchurDecomposition(true).decompose(src);
        CMatrix U = schur.getU();

        if(src.isHermitian()) {
            // Then the columns of U are the complete orthonormal set of eigenvectors of the src matrix.
            return U;
        } else {
            // For a non-symmetric matrix, only the first column of U will be an eigenvector of the src matrix.
            CMatrix Q = getEigenVectorsTriu(schur.getT()); // Compute the eigenvectors of T.
            return U.mult(Q); // Convert the eigenvectors of T to the eigenvectors of the src matrix.
        }
    }


    /**
     * Computes the eigenvectors of an upper triangular matrix.
     * @param T The upper triangular matrix to compute the eigenvectors of.
     * @return A matrix containing the eigenvectors of {@code T} as its columns.
     */
    public static CMatrix getEigenVectorsTriu(Matrix T) {
        RealBackSolver backSolver = new RealBackSolver();
        Matrix I = Matrix.I(T.numRows);
        CMatrix Q = new CMatrix(T.numRows);

        for(int j=0; j<T.numRows; j++) {
            Matrix t = I.mult(T.get(j, j)); // TODO: add diag(...) method so this scalar multiplication doesnt need to be computed.
            Matrix S = T.sub(t).getSlice(0, j+1, 0, j+1);
            Matrix S_hat = S.getSlice(0, j, 0, j);
            Vector r = S.getSlice(0, j, S.numCols-1, S.numCols).toVector();
            Vector v;

            if(S_hat.entries.length > 0) {
                v = backSolver.solve(S_hat, r.mult(-1));
                v = v.join(new Vector(1.0));
            } else {
                v = new Vector(1.0);
            }

            v = v.normalize().join(new Vector(T.numRows-v.size));
            Q.setCol(v.entries, j);
        }

        return Q;
    }


    /**
     * Computes the eigenvectors of an upper triangular matrix.
     * @param T The upper triangular matrix to compute the eigenvectors of.
     * @return A matrix containing the eigenvectors of {@code T} as its columns.
     */
    public static CMatrix getEigenVectorsTriu(CMatrix T) {
        ComplexBackSolver backSolver = new ComplexBackSolver();
        Matrix I = Matrix.I(T.numRows);
        CMatrix Q = new CMatrix(T.numRows);

        for(int j=0; j<T.numRows; j++) {
            CMatrix t = I.mult(T.get(j, j)); // TODO: add diag(...) method so this scalar multiplication doesnt need to be computed.
            CMatrix S = T.sub(t).getSlice(0, j+1, 0, j+1);
            CMatrix S_hat = S.getSlice(0, j, 0, j);
            CVector r = S.getSlice(0, j, S.numCols-1, S.numCols).toVector();
            CVector v;

            if(S_hat.entries.length > 0) {
                v = backSolver.solve(S_hat, r.mult(-1));
                v = v.join(new CVector(1.0));
            } else {
                v = new CVector(1.0);
            }

            v = v.normalize().join(new Vector(T.numRows-v.size));
            Q.setCol(v, j);
        }

        return Q;
    }


    /**
     * Computes the eigenvalues and eigenvectors of a square real matrix.
     * @param src The matrix to compute the eigenvalues and vectors of.
     * @return An array containing two matrices. The first matrix has shape {@code 1xsrc.numCols} and contains the eigenvalues
     * of the {@code src} matrix. The second matrix has shape {@code src.numRowsxsrc.numCols} and contains the
     * eigenvectors of the {@code src} matrix as its columns.
     */
    public static CMatrix[] getEigenPairs(Matrix src) {
        CMatrix lambdas = new CMatrix(1, src.numRows);

        SchurDecomposition<Matrix, Vector> schur = new RealSchurDecomposition(true).decompose(src);
        CMatrix T = schur.getT();
        CMatrix U = schur.getU();

        // Extract diagonal of T.
        for(int i=0; i<T.numRows; i++) {
            lambdas.entries[i] = T.entries[i*(T.numCols + 1)];
        }

        if(src.isSymmetric()) {
            // Then the columns of U are the complete orthonormal set of eigenvectors of the src matrix.
            return new CMatrix[]{lambdas, U};
        } else {
            // For a non-symmetric matrix, only the first column of U will be an eigenvector of the src matrix.
            U = U.mult(getEigenVectorsTriu(T)); // Compute the eigenvectors of T and convert to eigenvectors of src.
            return new CMatrix[]{lambdas, U};
        }
    }


    /**
     * Computes the eigenvalues and eigenvectors of a square real matrix.
     * @param src The matrix to compute the eigenvalues and vectors of.
     * @return An array containing two matrices. The first matrix has shape {@code 1xsrc.numCols} and contains the eigenvalues
     * of the {@code src} matrix. The second matrix has shape {@code src.numRowsxsrc.numCols} and contains the
     * eigenvectors of the {@code src} matrix as its columns.
     */
    public static CMatrix[] getEigenPairs(CMatrix src) {
        CMatrix lambdas = new CMatrix(1, src.numRows);

        SchurDecomposition<CMatrix, CVector> schur = new ComplexSchurDecomposition(true).decompose(src);
        CMatrix T = schur.getT();
        CMatrix U = schur.getU();

        // Extract diagonal of T.
        for(int i=0; i<T.numRows; i++) {
            lambdas.entries[i] = T.entries[i*(T.numCols + 1)];
        }

        if(src.isHermitian()) {
            // Then the columns of U are the complete orthonormal set of eigenvectors of the src matrix.
            return new CMatrix[]{lambdas, U};
        } else {
            // For a non-symmetric matrix, only the first column of U will be an eigenvector of the src matrix.
            U = U.mult(getEigenVectorsTriu(T)); // Compute the eigenvectors of T and convert to eigenvectors of src.
            return new CMatrix[]{lambdas, U};
        }
    }
}
