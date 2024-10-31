import S3UploadForm from "./components/S3UploadForm";
import styles from "./page.module.css";

export default function Home() {
  return (
    <div className={styles.page}>
      <S3UploadForm />
    </div>
  );
}
