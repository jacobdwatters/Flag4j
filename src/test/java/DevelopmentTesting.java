import com.flag4j.Matrix;
import com.flag4j.util.RandomTensor;


public class DevelopmentTesting {


    public static Matrix standardT(Matrix A) {
        Matrix t = new Matrix(A.numCols(), A.numRows());

        for(int i=0; i<t.entries.length; i++) {
            for(int j=0; j<t.entries[0].length; j++) {
                t.entries[i][j] = A.entries[j][i];
            }
        }

        return t;
    }


    public static void main(String[] args) {
        RandomTensor rand = new RandomTensor();

        int numRows = 1000;
        int numCols = 1000;
        Matrix A = rand.getRandomMatrix(numRows, numCols);

        final long startTime = System.currentTimeMillis();
        standardT(A);
        final long endTime = System.currentTimeMillis();

        System.out.println("Total execution time: " + (endTime - startTime));
    }
}
