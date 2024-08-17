package org.flag4j.vector;

import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.dense.TensorOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.core.Shape;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class VectorConversionTests {

    double[] aEntries;
    VectorOld a;

    double[][] matExpEntries;
    MatrixOld matExp;

    double[] tensorExpEntries;
    Shape tensorExpShape;
    TensorOld tensorExp;

    @Test
    void toMatrixTestCase() {
        // --------------------- Sub-case 1  ---------------------
        aEntries = new double[]{1.34, -8.244, 1.234, 90031.3};
        a = new VectorOld(aEntries);
        matExpEntries = new double[][]{{1.34}, {-8.244}, {1.234}, {90031.3}};
        matExp = new MatrixOld(matExpEntries);

        assertEquals(matExp, a.toMatrix());

        // --------------------- Sub-case 2  ---------------------
        aEntries = new double[]{1.34, -8.244, 1.234, 90031.3};
        a = new VectorOld(aEntries);
        matExpEntries = new double[][]{{1.34}, {-8.244}, {1.234}, {90031.3}};
        matExp = new MatrixOld(matExpEntries);

        assertEquals(matExp, a.toMatrix(true));

        // --------------------- Sub-case 2  ---------------------
        aEntries = new double[]{1.34, -8.244, 1.234, 90031.3};
        a = new VectorOld(aEntries);
        matExpEntries = new double[][]{{1.34, -8.244, 1.234, 90031.3}};
        matExp = new MatrixOld(matExpEntries);

        assertEquals(matExp, a.toMatrix(false));
    }


    @Test
    void toTensorTestCase() {
        // --------------------- Sub-case 1  ---------------------
        aEntries = new double[]{1.34, -8.244, 1.234, 90031.3};
        a = new VectorOld(aEntries);
        tensorExpEntries = new double[]{1.34, -8.244, 1.234, 90031.3};
        tensorExpShape = new Shape(aEntries.length);
        tensorExp = new TensorOld(tensorExpShape, tensorExpEntries);

        assertEquals(tensorExp, a.toTensor());

        // --------------------- Sub-case 2  ---------------------
        aEntries = new double[]{-34534, 6.2234};
        a = new VectorOld(aEntries);
        tensorExpEntries = new double[]{-34534, 6.2234};
        tensorExpShape = new Shape(aEntries.length);
        tensorExp = new TensorOld(tensorExpShape, tensorExpEntries);

        assertEquals(tensorExp, a.toTensor());
    }
}
